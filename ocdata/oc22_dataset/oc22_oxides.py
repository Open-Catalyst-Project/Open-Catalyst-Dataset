import os, glob, random, time, json, logging, argparse, copy, string, itertools, pickle

from pymatgen.core.structure import *
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from json.decoder import JSONDecodeError

from ocdata.oc22_dataset.MXide_adsorption import MXideAdsorbateGenerator
from ocdata.oc22_dataset.sets import MOSurfaceSet, set_bulk_magmoms
from ocdata.oc22_dataset.termination_generator import get_random_clean_slabs, center_slab
from ocdata.oc22_dataset.adsorbate_configs import adslist, OOH_list


def rid_generator(r=random):
    return ''.join([r.choice(string.ascii_letters
                             + string.digits) for n in range(10)])


class OC22:
    """
    A class that selects metal-oxide adsorbate/slab objects and writes vasp 
        input files for all materials available in the list of config files. 
        Not when inputting n_slabs, n_ads and n_adslabs, the total number of 
        calculations (vasp directories) is a sequential product of those three
        numbers. ie for n_slabs=1, n_ads=2 and n_adslabs=3 means we select one 
        random facet/termination and create two sets of adsorbed slabs with 
        said termination from two randomly selected adsorbates, each set having 
        3 adslabs with a randomly selected xyz, rotation and coverage of the 
        adsorbate. In total this input will generate 7 vasp folders, 6 adslab 
        calculations and 1 clean termination calculation.
        
    .. attribute:: master_folder
        
        Name of folder containing all calculations. Name typically 
            contains the seed used for randomly generating inputs

    """
    def __init__(self, output_dir, n_slabs, n_adslabs, n_ads, total_calcs, seed, 
                 max_index=3, primitive=False, max_normal_search=None,
                 bulk_db='MP_unary_binary_oxides.json', n_rotations=4,
                 rot_axis=[0,0,1], min_lw=8.0, repeat=None, height=1.25, 
                 max_coverage=None, limit_bulk_atoms=72, limit_slab_atoms=300, user_incar_settings={}, 
                 slabgen_inputs={}, verbose=False, fail_safe=True):
        
        """
        output_dir (str): The output directory structure will have a based directory for
            each material with the reduced formula followed by the MPID. e.g. "VO2_mp-19094".
            Inside this directory will be a list of directories containing the VASP input
            files needed for calculations. The name of the directories for clean and adsorbed
            slabs are derived from the keys of the struct_dict using the MOxideGenerator.
        n_slabs (int): N number of random clean slabs
        n_ads (int): N number of random adsorbates
        n_adslabs (int): N number of random adsorbate configurations for each
            randomly selected clean slab
        total_calcs (int): N number of adslab calculations to generate
        bulk_db (str): Directory leading to the json files containing 
            the bulk ComputedStructureEntry objects as dicts
        max_index (int): The maximum index. For example, a max_index of 1 means that
            (100), (110), and (111) are returned for the cubic structure. All other
            indices are equivalent to one of these.
        seed (int): Random seed for sampling
        user_incar_settings (dict): User INCAR settings. This allows a user
            to override INCAR settings, e.g., setting a different MAGMOM for
            various elements or species. Note that in the new scheme,
            ediff_per_atom and hubbard_u are no longer args. Instead, the
            config_dict supports EDIFF_PER_ATOM and EDIFF keys. The former
            scales with # of atoms, the latter does not. If both are
            present, EDIFF is preferred. To force such settings, just supply
            user_incar_settings={"EDIFF": 1e-5, "LDAU": False} for example.
            The keys 'LDAUU', 'LDAUJ', 'LDAUL' are special cases since
            pymatgen defines different values depending on what anions are
            present in the structure, so these keys can be defined in one
            of two ways, e.g. either {"LDAUU":{"O":{"Fe":5}}} to set LDAUU
            for Fe to 5 in an oxide, or {"LDAUU":{"Fe":5}} to set LDAUU to
            5 regardless of the input structure.

        See parse_args() for a full description of all parameters.

        Public methods:
        run()
            selects the appropriate materials and writes to files
        """

        # inputs for generating clean slabs
        if not slabgen_inputs:
            self.slabgen_inputs = {"min_slab_size": 10, "min_vacuum_size": 20,
                                   "lll_reduce": False, "in_unit_planes": False, 
                                   "primitive": primitive, "max_normal_search": max_normal_search}
        else:
            self.slabgen_inputs = slabgen_inputs
        self.max_index = max_index

        # inputs for random sampling
        self.n_slabs = n_slabs
        self.n_adslabs = n_adslabs
        self.n_ads = n_ads
        self.seed = seed
        random.seed(self.seed)
        self.random = random
        self.total_calcs = total_calcs

        # inputs for adsorption generator
        self.n_rotations = n_rotations
        self.rot_axis = rot_axis
        self.min_lw = min_lw
        self.repeat = repeat
        self.height = height
        self.max_coverage = max_coverage
        self.limit_bulk_atoms = limit_bulk_atoms
        self.limit_slab_atoms = limit_slab_atoms
        
        # DB for querying bulks
        self.bulk_db = bulk_db
                
        # vasp inputs
        self.user_incar_settings = user_incar_settings
        print('user_incar_settings', self.user_incar_settings)
        print('output_dir', output_dir)
        self.output_dir = output_dir

        self.fail_safe = fail_safe
        self.verbose = verbose
        
        # initialize proper input sets, this is a work around 
        # the default pmg settings that needs to be fixed in pmg later
        mvl = MOSurfaceSet(Structure(Lattice.cubic(4), ['Fe'], [(0,0,0)]))
        
    def get_adsorbate_width(self, molecule):
        """
        Get the maximum width of the adsorbate in order evenly space out multiple adsorbates
        """

        adsorbate = molecule[0] if type(molecule).__name__ == 'list' else molecule.copy()
        if len(adsorbate) == 1:
            return 0
        else:
            all_dists = []
            for pair_site in itertools.combinations(adsorbate, 2):
                i = adsorbate.index(pair_site[0])
                j = adsorbate.index(pair_site[1])
                all_dists.append(adsorbate.get_distance(i, j))
            return max(all_dists)

    def get_vasp_set(self, slab):
        auto_dipole = True if 'adsorbate' in slab.site_properties['surface_properties'] else False
        return MOSurfaceSet(slab.copy(), **{'user_incar_settings': self.user_incar_settings,
                                            'auto_dipole': auto_dipole})

    def _load_adsorbates(self):

        # Get adsorbates
        ads_todo = copy.copy(adslist)
        ads_todo.append(0)
        
        ads_todo = self.random.sample(ads_todo, self.n_ads)

        molecules = []
        for m in ads_todo:
            if type(m).__name__ == 'int':
                molecules.append(self.random.sample(OOH_list, 1)[0])
            else:
                molecules.append(m)

        return molecules
    
    def get_O_vac_coverage(self, slab):
        """
        The coverage of excess or defficient oxygen atoms at the surface. Excess oxygen 
            occurs when there are more oxygens than metal atoms per formula unit. Note that
            metal formula units will be treated as the same value for all metals. ie for 
            FeCu2O3, number M=3 and number O=3/ Coverage is defined per unit surface since
            both terminations are symmetrically equivalent. i.e. divide by 2. Coverage < 0
            represents an O defficient system, = 0 is stoichiometric and > 0 is O excess.
        """
        slab_comp = slab.composition.as_dict()
        bulk_comp = slab.oriented_unit_cell.composition.reduced_composition.as_dict()
        nM = sum([bulk_comp[el] for el in bulk_comp.keys() if el != 'O'])
        nM_slab = sum([slab_comp[el] for el in slab_comp.keys() if el != 'O'])
        nexcess = slab_comp['O'] - bulk_comp['O']*(nM_slab/nM)
        return nexcess/(slab.surface_area*2)
        
    def run(self):
        """
        Generates random directories containing VASP inputs for all clean and adsorbed slabs.
        """
        start = time.time()
        
        # load the bulk structures
        all_pbx_stable_entries = [ComputedStructureEntry.from_dict(d) for d in \
                                  json.load(open(self.bulk_db))]
                
        count = 0
        tot_start = time.time()
        
        # generate all vasp directories starting from a random sampling of a bulk
        while count < self.total_calcs:

            # get a random bulk entry
            entry = self.random.sample(all_pbx_stable_entries, 1)[0]
            sg = SpacegroupAnalyzer(entry.structure)
            if self.verbose:
                print('Entry ID: ', entry.entry_id, entry.composition.reduced_formula, 
                      '%s atoms' %(len(entry.structure)), sg.get_space_group_symbol(), 
                      sg.get_crystal_system())
                
            # set the magmom of the bulk based on CFT
            bulk = entry.structure.copy()
            if len(bulk) > self.limit_bulk_atoms:
                continue

            # get random clean slabs
            slab_start = time.time()
            mmi = 2 if sg.get_crystal_system() in ['triclinic', 'monoclinic'] else self.max_index
            mo_slabs_selected = get_random_clean_slabs(bulk, self.n_slabs, mmi, 
                                                       min_lw=self.min_lw, r=self.random,
                                                       **self.slabgen_inputs)
            if not mo_slabs_selected:
                print('NO VIABLE CLEAN SLABS')
                continue
                                
            slab_end = time.time()
            print('Sampled %s clean slabs in %.3f' %(len(mo_slabs_selected), slab_end-slab_start))
            # now get random adsorbed slabs
            adslab_start = time.time()
            struct_dict = {}
            for slab in mo_slabs_selected:
                slab_id = rid_generator(self.random)

                comp = entry.composition.reduced_formula
                if '(' in comp:
                    comp = comp.replace('(', '-')
                    comp = comp.replace(')', '-')
                
                clean_label = '%s_%s_clean_%s' % (comp, entry.entry_id, slab_id)
                
                mxide_adsgen = MXideAdsorbateGenerator(slab, height=self.height, 
                                                       repeat=[1,1,1], r=self.random)

                struct_dict.update({clean_label: mxide_adsgen.slab.copy()})

                # Get random adsorbates
                if len(mxide_adsgen.MX_adsites) == 0:
                    adsset = [molecule for molecule in self._load_adsorbates() 
                              if type(molecule).__name__ != 'list']
                    nads = len(adsset) if self.n_ads > len(adsset) else self.n_ads
                    random_molecules = self.random.sample(adsset, nads)
                else:
                    adsset = self._load_adsorbates()
                    random_molecules = self.random.sample(adsset, self.n_ads)    

                # Get random sampling of adsorbed slabs, n_adslabs per adsorbate
                for molecule in random_molecules:

                    if self.get_adsorbate_width(molecule) > \
                    ((mxide_adsgen.min_adsorbate_dist) / 2) * (1.15):
                        cov = 1
                    elif hasattr(molecule, 'dimer_coupling'):
                        # just do a coverage of 1 dimer couple per surface for now
                        cov = 1
                        if len(mxide_adsgen.surf_metal_sites) < 2:
                            continue
                    else: 
                        cov = self.max_coverage
                
                    adslabs = mxide_adsgen. \
                        generate_random_adsorption_structure(molecule, self.n_adslabs,
                                                             max_coverage=cov, axis=self.rot_axis, 
                                                             n_rotations=self.n_rotations)
                    if not adslabs:
                        if hasattr(molecule, 'dimer_coupling'):
                            adslabs = mxide_adsgen. \
                                generate_random_adsorption_structure(self.random.sample(adsset, 1)[0],
                                                                     self.n_adslabs, max_coverage=cov, 
                                                                     axis=self.rot_axis, 
                                                                     n_rotations=self.n_rotations)
                            if not adslabs:
                                continue
                        else:
                            continue
                    
                    for adslab in adslabs:
                        # skip slabs with more than limit_slab_atoms atoms
                        if len(adslab) > self.limit_slab_atoms:
                            print('adslab too large (%s)!' %(len(adslab)))
                            continue
                        if self.fail_safe:
                            try:
                                s = Structure(adslab.lattice, adslab.species, adslab.frac_coords, 
                                              validate_proximity=True)
                                for i, site in enumerate(adslab):
                                    if site.surface_properties == 'adsorbate':
                                        if any([site.distance(new_site) < 0.75 
                                                for ii, new_site in enumerate(adslab) if i != ii]):
                                            raise StructureError("Adsorbates 0.75Å close to other atoms!")

                            except StructureError:
                                print('overlapping atoms in adslab, skipping!!!')
                                continue

                        adslab_id = rid_generator(self.random)
                        label = '%s_%s' % (clean_label.replace('clean_', ''), adslab_id)
                        setattr(adslab, 'adsorbate', molecule)
                        struct_dict.update({label: adslab})

            adslab_end = time.time()
            if self.verbose:
                print('Slab: %.3f sec Adslabs: %.3f sec' %(slab_end - slab_start, adslab_end - adslab_start))

            # skip if no adsorbed slabs
            if len([k for k in struct_dict.keys() if 'clean' not in k]) == 0:
                continue

            # finally, generate vasp inputs for slab and adslab
            for k in struct_dict.keys():

                mvlset = self.get_vasp_set(struct_dict[k])
                # need to initialize the mvlset before writing, need to fix in pmg
                mvlset.potcar_functional
                if self.fail_safe:
                    # Check if potcar compatible with PBE_54
                    potcars = mvlset.potcar
                    for p in potcars:
                        if 'PBE_54' not in p.identify_potcar()[0]:
                            if self.verbose:
                                print('POTCAR INCOMPATIBLE WITH PBE_54, SKIPPING %s'  %(k))

                            continue
                if self.verbose:
                    print(k)

                mvlset.write_input(os.path.join(self.output_dir, self.master_folder, k))

                # make inputs pkls for record keeping
                inputs_dict = {"INCAR": mvlset.incar.as_dict(), "POSCAR": mvlset.poscar.as_dict(), 
                               "POTCAR": mvlset.potcar.as_dict(), "KPOINTS": mvlset.kpoints.as_dict(),
                               "slab": struct_dict[k].as_dict()}
                with open(os.path.join(self.output_dir, self.master_folder, k, 
                                       'inputs.pkl'), 'wb') as outfile:
                    pickle.dump(inputs_dict, outfile)
                outfile.close()

                # make generator pkls for record keeping
                generator = {'trajectory': struct_dict[k]}
                generator['tags'] = struct_dict[k].site_properties['surface_properties']

                # clean slab data
                if 'clean' in k:
                    clean_slab = struct_dict[k] 
                    generator['clean_slab_id'] = k
                else:
                    comp, entry.entry_id, clean_id, ads_id = k.split('_')
                    generator['clean_slab_id'] = '%s_%s_clean_%s' %(comp, entry.entry_id, clean_id)
                    clean_slab = struct_dict['%s_%s_clean_%s' %(comp, entry.entry_id, clean_id)]
                generator['entry_id'] = entry.entry_id
                generator['miller_index'] = clean_slab.miller_index
                generator['shift'] = clean_slab.shift
                generator['O_coverage'] = self.get_O_vac_coverage(clean_slab)

                # ads slab data
                if 'clean' not in k:
                    generator['adslab_id'] = k
                    adslab = struct_dict[k] 
                    mol = adslab.adsorbate[0] if type(adslab.adsorbate).__name__ == 'list' else adslab.adsorbate
                    if 'dimer_coupling' in mol.site_properties.keys():
                        generator['adsorbate_name'] = '%s_couple' %(mol.composition.reduced_formula)
                    elif len(mol) == 1:
                        generator['adsorbate_name'] = mol[0].species_string
                    else:
                        generator['adsorbate_name'] = mol.composition.reduced_formula
                    generator['adsorbate_molecule'] = mol
                    nads = len([site for site in adslab if site.surface_properties == 'adsorbate'])/len(mol)
                    generator['ads_coverage'] = '%.3f' %(nads/adslab.surface_area)
                    generator['adsorbate_sites'] = sorted([[round(c,3) for c in coord] for coord in adslab.ads_coords])

                    # keep track of how many adslabs inputs made
                    count+=1

                with open(os.path.join(self.output_dir, self.master_folder, k, 'generator.pkl'), 'wb') as outfile:
                    pickle.dump(generator, outfile)
                outfile.close()

                if count == self.total_calcs:
                    break
                                            
        tot_end = time.time()
        if self.verbose:
            print('Time analysis:')
            print('Total: %.3f sec' %(tot_end - tot_start))
                    
        end = time.time()
        print('Elapsed time for %s inputs: %s sec' %(count, end-start))
        
        
def parse_args():
    
    parser = argparse.ArgumentParser(description='Generate VASP inputs for clean and adsorbed metal-oxide slabs')

    # Required parameters for sampling clean and adsorbed slabs
    parser.add_argument('--n_slabs', type=int, required=True, help='Number of clean slabs to sample')
    parser.add_argument('--n_ads', type=int, required=True, help='Number of adsorbates to sample')
    parser.add_argument('--n_adslabs', type=int, required=True,
                        help='Number of adsorbed slab configurations per adsorbate to sample')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for sampling')
    parser.add_argument('--total_calcs', type=int, required=True, help='Total number of adslab vasp inputs to make')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to write inputs to')

    # Optional parameters for generating clean slabs
    parser.add_argument('--max_index', type=int,default=3, help='Max Miller index of slabs')
    parser.add_argument('--limit_bulk_atoms', type=int,default=72, help='Max number of atoms in bulk')
    parser.add_argument('--limit_slab_atoms', type=int,default=300, help='Max number of atoms in slab')
    parser.add_argument('--primitive', type=bool, default=None, help='Reduce slabs to smallest size')
    parser.add_argument('--max_normal_search', type=int,default=None, help='For getting orthogonal c')

    # Optional parameters for configuring adsorbates on the surface
    parser.add_argument('--n_rotations', type=int, default=4,
                        help='Incremental number of rotations of adsorbate about rot_axis')
    parser.add_argument('--rot_axis', type=list, default=[0,0,1], help='Axis of adsorbate to rotate about')
    parser.add_argument('--min_lw', type=float, default=8.0, help='Minimum width/length of surface (Å)')
    parser.add_argument('--repeat', type=list, default=None, help='Slab supercell to accomadate adsorbate')
    parser.add_argument('--height', type=float, default=1.25, help='Surface sites height considered from top of slab')
    parser.add_argument('--max_coverage', type=int, default=None, help='Max number of adsorbates on surface')
    
    # DB for querying bulks
    parser.add_argument('--bulk_db', type=str, default=os.path.join(os.getcwd(), 'bulk_oxides.json'), 
                        required=False, help='Json file of bulk pmg ComputedStructureEntries')    
    
    # Optional parameters for VASP input set
    parser.add_argument('--user_incar_settings', type=str, default='{}', help='User INCAR settings')
    parser.add_argument('--verbose', type=bool, default=False, help='Print long messages')    
    
    # check that all needed args are supplied
    args = parser.parse_args()
    if args.output_dir is None or args.n_slabs is None or args.n_ads is None or args.n_adslabs is None:
        parser.error('Enumerating all structures requires specified n_slabs, n_ads, n_adslabs and output_dir')
    return args

if __name__ == '__main__':

    args = parse_args()
    job = OC22(**{"output_dir": args.output_dir, "n_slabs": args.n_slabs, "n_ads": args.n_ads,
                  "n_adslabs": args.n_adslabs, "max_index": args.max_index, "total_calcs": args.total_calcs,
                  "n_rotations": args.n_rotations, "rot_axis": args.rot_axis, "min_lw": args.min_lw,
                  "repeat": args.repeat, "height": args.height, "max_coverage": args.max_coverage,
                  "seed": args.seed, "user_incar_settings": json.loads(args.user_incar_settings),
                  'limit_bulk_atoms': args.limit_bulk_atoms, 'bulk_db': args.bulk_db, 
                  'limit_slab_atoms': args.limit_slab_atoms, 'verbose': args.verbose, 
                  'primitive': args.primitive, 'max_normal_search': args.max_normal_search})
    job.run()
    
    