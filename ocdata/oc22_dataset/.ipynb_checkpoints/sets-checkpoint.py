import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core.periodic_table import Specie

"""
All credits for this python module goes to:

@article{wherewulff2022,
    title = {WhereWulff: An Autonomous Workflow to Democratize and Scale Complex Material Discovery for Electocatalysis},
    author = {Rohan Yuri Sanspeur, Javier Heras-Domingo and Zachary Ulissi},
    journal = {in preparation},
    year = {2022}}
"""


# Get bulk initial magnetic moments
def set_bulk_magmoms(structure, tol=0.1, scale_factor=1.2):
    """
    Returns decorated bulk structure with initial magnetic moments,
    based on crystal-field theory for TM.
    """
    struct = structure.copy()
    # Voronoi NN
    voronoi_nn = VoronoiNN(tol=tol)
    # SPG Analysis
    sga = SpacegroupAnalyzer(struct)
    sym_struct = sga.get_symmetrized_structure()
    # Magnetic moments
    element_magmom = {}
    for idx in sym_struct.equivalent_indices:
        site = sym_struct[idx[0]]
        if site.specie.is_transition_metal:
            
            elec = site.specie.full_electronic_structure
            if len(elec) < 4 or elec[-1][1] != "s" or elec[-2][1] != "d":
                return structure
            nelectrons = elec[-1][2] + elec[-2][2] - site.specie.oxi_state
            if nelectrons < 0 or nelectrons > 10:
                return structure
            
            cn = voronoi_nn.get_cn(sym_struct, idx[0], use_weights=True)
            cn = round(cn, 5)
            # Filter between Oh or Td Coordinations
            if cn > 5.0:
                coordination = "oct"
            else:
                coordination = "tet"
            # Spin configuration depending on row
            if site.specie.row >= 5.0:
                spin_config = "low"
            else:
                spin_config = "high"
            # Magnetic moment per metal site
            magmom = Specie(site.specie).get_crystal_field_spin(
                coordination=coordination, spin_config=spin_config
            )
            # Add to dict
            element_magmom.update(
                {str(site.specie.name): abs(scale_factor * float(magmom))}
            )

        elif site.specie.is_chalcogen:  # O
            element_magmom.update({str(site.specie.name): 0.6})

        else:
            element_magmom.update({str(site.specie.name): 0.0})

    magmoms = [element_magmom[site.specie.name] for site in struct]

    # Decorate
    for site, magmom in zip(struct.sites, magmoms):
        site.properties["magmom"] = magmom
    return struct

# Help function to avoid tuples as key.
def json_format(dt):
    dt_new = {}
    for k, v in dt.items():
        k_str = "".join(map(str, k))
        dt_new.update({k_str: v})
    return dt_new

# Theoretical Level
class MOSurfaceSet(MVLSlabSet):
    """
    Custom VASP input class for MO slab calcs
    """
    def __init__(self, structure, bulk=False, apply_U=True, set_mix=False, 
                 psp_version="PBE_54", auto_dipole=True, **kwargs):

        super(MOSurfaceSet, self).__init__(
            structure, bulk=bulk, set_mix=set_mix, 
            auto_dipole=auto_dipole, **kwargs)
        
        config_file = MOSurfaceSet.CONFIG
        self.psp_version = psp_version
        # Change default PBE version from PMG
        psp_versions = ['PBE', 'PBE_52', 'PBE_54'] #  'LDA', 'PW91' 
        assert self.psp_version in psp_versions
#         config_file['POTCAR_FUNCTIONAL'] = self.psp_version
        
        self.bulk = bulk
        self.auto_dipole = auto_dipole    
    
        # Make it compatible with PBE_54 POTCAR symbols
        self.CONFIG['POTCAR']['W'] = 'W_sv'
        self._config_dict['POTCAR']['W'] = 'W_sv'
#         config_file = MOSurfaceSet.CONFIG

    @property
    def incar(self):
        
        incar = super(MOSurfaceSet, self).incar

        # Direct of reciprocal (depending if its bulk or slab)
        if self.bulk:
            incar["LREAL"] = True
        else:
            incar["LREAL"] = False

        # Setting auto_dipole correction (for slabs only)
        #if not self.bulk and self.auto_dipole:
        #    incar["LDIPOL"] = True
        #    incar["IDIPOL"] = 3
        #    incar["DIPOL"] = self._get_center_of_mass()

        # Incar Settings for optimization
        incar_config = {
            "GGA": "PE",
            "ENCUT": 500,
            "EDIFF": 1e-4,
            "EDIFFG": -0.05,
            "ISYM": 0,
            "SYMPREC": 1e-10,
            "ISPIN": 2,
            "ISIF": 0,
            "NSW": 300,
            "NCORE": 4,
            "LWAVE": True,
            "ISTART": 1,
            "NELM": 60
        }
        # Update incar
        incar.update(incar_config)
        incar.update(self.user_incar_settings)
        if self.auto_dipole:
            incar['DIPOL'] = '%.10f, %.10f %.10f' %tuple(incar['DIPOL'])
        
        return incar
    
    @property
    def kpoints(self):
        """
        Monkhorst-pack Gamma Centered scheme:
            bulks [50/a x 50/b x 50/c]
            slabs [30/a x 30/b x 1]
        """
        abc = np.array(self.structure.lattice.abc)

        if self.bulk:
            kpts = tuple(np.ceil(50.0 / abc).astype('int'))
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0,0,0))

        else:
            kpts = np.ceil(30.0 / abc).astype('int')
            kpts[2] = 1
            kpts = tuple(kpts)
            return Kpoints.gamma_automatic(kpts=kpts, shift=(0,0,0))
