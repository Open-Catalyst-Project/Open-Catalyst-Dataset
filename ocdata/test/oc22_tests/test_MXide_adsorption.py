import os, unittest, sys, json
# cwd = os.getcwd()
# sys.path.append(cwd.replace(cwd.split('/')[-1], ''))

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Structure, Lattice, Molecule
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedStructureEntry

from ocdata.oc22_dataset.surface import SlabGenerator, generate_all_slabs, Slab
from ocdata.oc22_dataset.adsorbate_configs import adslist, OOH_list
from ocdata.oc22_dataset.MXide_adsorption import MXideAdsorbateGenerator
from ocdata.oc22_dataset.termination_generator import get_random_clean_slabs

__author__ = "Richard Tran"

class MXideAdsorbateGeneratorTest(PymatgenTest):
    def setUp(self):

        # Test on a generic rutile structure
        l = Lattice.from_parameters(4.65327, 4.65327, 2.96920, 90, 90, 90)
        species = ['Ti', 'Ti', 'O', 'O', 'O', 'O']
        fcoords = [[0.5, 0.5, 0.5], [0, 0, 0], [0.19542, 0.80458, 0.5],
                   [0.80458, 0.19542, 0.5], [0.30458, 0.30458, 0], [0.69542, 0.69542, 0]]
        self.rutile = Structure(l, species, fcoords)
        slabgen = SlabGenerator(self.rutile, (1, 1, 0), 4, 8, lll_reduce=True,
                                center_slab=True, in_unit_planes=True)
        # generates 3 slabs, a stoiochiometric, a nonstoichiometric oxygen
        # excess, and a nonstoichiometric oxygen defficient slab
        slabs = slabgen.get_slabs(symmetrize=True)
        self.bondlength = 2.20
        slabdict = {}
        for slab in slabs:
            d = slab.composition.as_dict()
            if d['Ti'] == d['O'] / 2:
                slabdict['stoichiometric'] = slab
            elif d['Ti'] < d['O'] / 2:
                slabdict['oexcess'] = slab
            elif d['Ti'] > d['O'] / 2:
                slabdict['odefficient'] = slab
        self.slabdict = slabdict
        
        self.adslist = adslist
        self.adslist.extend(OOH_list)
        self.Nitro_coupling = Molecule(["N", "N"], [(1.6815, 0, 0,), (0,0,0)])
        self.Nitro_coupling.add_site_property('dimer_coupling', [True, True])
        self.mo_prototypes = [ComputedStructureEntry.from_dict(d) \
                              for d in json.load(open('mo_prototypes.json', 'rb'))]
        # will set self.long to False by default. If true, we
        # attempt to build random slabs from every single prototype MO
        self.long = False

    def test_site_finder(self):

        # There are two algorithms for finding adsorption sites:
            # 1. The adsorbate is located at the surface O-vacancy lattice position
            # 2. The adsorbate is bonded to an existing surface O in such a way that it forms a new molecule

        mxidegen = MXideAdsorbateGenerator(self.slabdict['stoichiometric'], repeat=[1, 1, 1])
        self.assertEqual(len(mxidegen.MX_adsites), 1)
        self.assertEqual(len(mxidegen.mvk_adsites), 1)

#         mxidegen = MXideAdsorbateGenerator(self.slabdict['oexcess'], repeat=[1, 1, 1])
#         self.assertEqual(len(mxidegen.MX_adsites), 0) # All surface M are fully coordinated
#         self.assertEqual(len(mxidegen.mvk_adsites), 2)

        mxidegen = MXideAdsorbateGenerator(self.slabdict['odefficient'], repeat=[1, 1, 1])
        self.assertEqual(len(mxidegen.MX_adsites), 2)
#         self.assertEqual(len(mxidegen.mvk_adsites), 3) # should be 2, need a better way of identifying surface sites

    def test_random_adslab_generation(self):

        for k in self.slabdict.keys():
            for molecule in self.adslist:
                mxidegen = MXideAdsorbateGenerator(self.slabdict[k], repeat=[1, 1, 1])
                if type(molecule).__name__ != 'list':
                    if 'dimer_coupling' in molecule.site_properties.keys():
                        if k == 'oexcess':
                            # can't make a dimer without metal sites
                            continue
                        metal_pos = [site.coords for site in mxidegen.slab if site.surface_properties == 'surface' 
                                     and site.species_string != mxidegen.X 
                                     and site.frac_coords[2] > mxidegen.slab.center_of_mass[2]]
                        count = 0
                        while len(metal_pos) < 5 and count < 3:
                            count+=1
                            mxidegen = MXideAdsorbateGenerator(self.slabdict[k], repeat=[1+count,1+count,1])
                            metal_pos = [site.coords for site in mxidegen.slab if site.surface_properties == 'surface' 
                                         and site.species_string != mxidegen.X 
                                         and site.frac_coords[2] > mxidegen.slab.center_of_mass[2]]                            

                # For adsorbing on existing O-sites, only MvK-like adsorption will work
                if k == 'oexcess' and type(molecule).__name__ != 'list':
                    continue
                try:
                    random_adslab = mxidegen.generate_random_adsorption_structure(molecule, 1)[0]
                except IndexError:
                    self.slabdict[k].to('cif', '%s.cif' %(k))
                    print('indexerrorindexerrorindexerrorindexerrorindexerrorindexerror', molecule)
                # Check if theres an adsorbate in the slab
                self.assertTrue('adsorbate' in random_adslab.site_properties['surface_properties'])
                s = Structure(random_adslab.lattice, random_adslab.species, 
                              random_adslab.frac_coords, validate_proximity=True)
                # now generate all slabs and check that the randomly sampled slab exists in the full set
                all_adslabs = mxidegen.generate_adsorption_structures(molecule, 'all')
#                 self.assertTrue(random_adslab in all_adslabs)            
                
    def test_adsorbate_saturation(self):
        
        all_slabs = generate_all_slabs(self.rutile, 3, 15, 15, lll_reduce=True,
                                       center_slab=True, symmetrize=True)
        Ox = Molecule(['O'], [(0,0,0)])
        for slab in all_slabs:
            mxidegen = MXideAdsorbateGenerator(slab, repeat=[1, 1, 1], adsorb_both_sides=True)
#             sat = mxidegen.generate_adsorption_structures(Ox, 'saturated')
#             if len(sat) == 0:
#                 # if no saturated slab returned, make sure its 
#                 # because theres no adsorption position found
#                 self.assertTrue(len(sat) == len(mxidegen.MX_adsites))
#             else:
#                 self.assertTrue(len(sat) == 1)
#                 s = Structure(sat[0].lattice, sat[0].species, sat[0].frac_coords, validate_proximity=True)
#                 self.assertTrue(sat[0].is_symmetric())
            
    def test_dimer_coupling(self):

        all_terms = get_random_clean_slabs(self.rutile, 5, 2, 15, 15)
        for i, s in enumerate(all_terms):
            slab = s.copy()
            mxidegen = MXideAdsorbateGenerator(slab, repeat=[1, 1, 1], height=1)
            metal_pos = [site for site in mxidegen.slab if site.surface_properties == 'surface' 
                         and site.species_string != mxidegen.X and site.frac_coords[2] > slab.center_of_mass[2]]
            if len(metal_pos) > 6:
                continue
            adslabs = mxidegen.generate_random_adsorption_structure(self.Nitro_coupling, 1, max_coverage=1)
            print('%s/5' %(i), len(adslabs))
            
    def test_relaxation(self):
        
        relax_slab = Slab.from_dict(json.load(open('relaxed_r110.json', 'rb')))
        unrelax_slab = Slab.from_dict(json.load(open('unrelaxed_r110.json', 'rb')))
        relax_mxidegen = MXideAdsorbateGenerator(relax_slab, positions=['MX_adsites'], relax_tol=0.125)
        unrelax_mxidegen = MXideAdsorbateGenerator(unrelax_slab, positions=['MX_adsites'])
#         self.assertTrue(len(unrelax_mxidegen.MX_adsites) == len(relax_mxidegen.MX_adsites))
        

if __name__ == "__main__":
    unittest.main()
