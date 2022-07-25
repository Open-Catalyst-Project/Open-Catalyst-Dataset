import os, unittest, sys, json, random
# cwd = os.getcwd()
# sys.path.append(cwd.replace(cwd.split('/')[-1], ''))

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.computed_entries import ComputedStructureEntry

from ocdata.oc22_dataset.MXide_adsorption import MXideAdsorbateGenerator
from ocdata.oc22_dataset.termination_generator import all_surface_site_combination, get_random_clean_slabs


__author__ = "Richard Tran"

class GenerateAllTermsTest(PymatgenTest):
    def setUp(self):

        # Test on a generic rutile structure
        self.rutile = Structure.from_file('RuO2.cif')
        slabgen = SlabGenerator(self.rutile, (1, 1, 0), 8, 8, lll_reduce=True, 
                                center_slab=True, in_unit_planes=True)
        # get the most O-saturated termination
        slabs = slabgen.get_slabs(symmetrize=True)
        Ocomp = 0
        satslab = None
        for slab in slabs:
            Ofrac = slab.composition.fractional_composition.as_dict()['O']
            if Ofrac > Ocomp:
                Ocomp = Ofrac
                satslab = slab
        mxide_gen = MXideAdsorbateGenerator(satslab, height=0.9, repeat=[1,1,1])
        self.satslab = mxide_gen.slab.copy()
        self.all_terms = all_surface_site_combination(self.satslab, [2,1,1], species=['O'])        
        
    def test_surface_symmetry(self):

        # make sure all terminations are symmetric
        self.assertTrue(all([s.is_symmetric() for s in self.all_terms]))
        # make sure all slabs symmetrically distinct 
        sm = StructureMatcher()
        self.assertEqual(len([g for g in sm.group_structures(self.all_terms)]), len(self.all_terms))
        # make sure to check proximity of all sites
        count = 0
        for slab in self.all_terms:
            try:
                s = Structure(slab.lattice, slab.species, 
                              slab.frac_coords, validate_proximity=True)
            except StructureError:
                count+=1
        self.assertEqual(count, 0)
        

class GenerateRandomTermsTest(PymatgenTest):
    def setUp(self):

        self.verbose = True
        # Test on a generic rutile structure
        self.rutile = Structure.from_file('RuO2.cif')
        
        self.all_terms = get_random_clean_slabs(self.rutile, 5, 3, 15, 15, min_lw=8.0, 
                                                lll_reduce=False, in_unit_planes=False, 
                                                primitive=True, max_normal_search=None)
        
        # will set self.long to False by default. If true, we
        # attempt to build random slabs from every single prototype MO
        self.long = False
        self.mo_prototypes = [ComputedStructureEntry.from_dict(d) \
                              for d in json.load(open('mo_prototypes.json', 'rb'))]
            
    def symmetry_proximity_check(self, all_terms):
        
        # make sure all terminations are symmetric
        self.assertTrue(all([s.is_symmetric() for s in all_terms]))
        
        # make sure to check proximity of all sites
        count = 0
        for slab in all_terms:
            try:
                s = Structure(slab.lattice, slab.species, slab.frac_coords, 
                              validate_proximity=True)
            except StructureError:
                count+=1
        self.assertEqual(count, 0)
        
    def test_surface_symmetry(self):
        
        # check symmetry of rutile
        self.symmetry_proximity_check(self.all_terms)
        # create a random slab for every prototype
        if self.long:
            for i, entry in enumerate(self.mo_prototypes):
                if self.verbose:
                    print('%s %s/%s' %(entry.entry_id, i, len(self.mo_prototypes)))
                s = entry.structure
                all_terms = get_random_clean_slabs(s, 3, 3, 15, 15, min_lw=1.0, 
                                                   lll_reduce=False, in_unit_planes=False, 
                                                   primitive=True, max_normal_search=None)
                self.symmetry_proximity_check(all_terms)
        else:
            entry = random.sample(self.mo_prototypes, 1)[0]
            if self.verbose:
                print('%s %s' %(entry.entry_id, len(self.mo_prototypes)))
            s = entry.structure
            all_terms = get_random_clean_slabs(s, 3, 3, 15, 15, min_lw=1.0, 
                                               lll_reduce=False, in_unit_planes=False, 
                                               primitive=True, max_normal_search=None)
            self.symmetry_proximity_check(all_terms)

            

if __name__ == "__main__":
    unittest.main()
