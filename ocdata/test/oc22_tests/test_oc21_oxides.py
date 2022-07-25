import os, unittest, sys, shutil, glob, filecmp
# cwd = os.getcwd()
# sys.path.append(cwd.replace(cwd.split('/')[-1], ''))
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.testing import PymatgenTest

from ocdata.oc22_dataset.MXide_adsorption import MXideAdsorbateGenerator
from ocdata.oc22_dataset.oc21_oxides import OC21
from ocdata.oc22_dataset.adsorbate_configs import adslist, OOH_list


__author__ = "Richard Tran"


class OC21Test(PymatgenTest):
    def setUp(self):

        # get all inputs for OC21 class. For this to truly be random, we will 
        # create an adslab from scratch (starting from the bulk) 50 times
        self.output_dir = os.path.join(cwd, 'vasp_test_folders')
        self.slabgen_inputs = {"min_slab_size": 15, "min_vacuum_size": 15,
                               "lll_reduce": False, "in_unit_planes": False, 
                               "primitive": True, "max_normal_search": 1}

    def test_run(self):
        """
        There are two algorithms for finding adsorption sites:
            1. The adsorbate is located at the surface O-vacancy lattice position
            2. The adsorbate is bonded to an existing surface O in such a way that it forms a new molecule
        """
        input_gen = OC21(self.output_dir, 1, 1, 1, 15, max_index=3, bulk_db='../bulk_oxides_20220621.json', 
                         seed=None, limit_bulk_atoms=50, verbose=True, slabgen_inputs=self.slabgen_inputs)
        input_gen.run()            
        self.cleanup(self.output_dir)
        
        # Make sure number of adslab folders made are consistent with total_calcs
        input_gen = OC21(self.output_dir, 3, 3, 3, 10, max_index=3, bulk_db='../bulk_oxides_20220621.json', 
                         seed=None, limit_bulk_atoms=50, verbose=True, slabgen_inputs=self.slabgen_inputs)
        input_gen.run()
        adscount = 0
        for f in glob.glob(os.path.join(self.output_dir, 'oc21_randomNone', '*')):
            if 'clean' not in f:
                adscount+=1
        self.assertEqual(adscount, 10)
        # Make sure every directory has inputs.pkl and generate.pkl
        for f in glob.glob(os.path.join(self.output_dir, 'oc21_randomNone', '*')):
            self.assertTrue(os.path.isfile(os.path.join(f, 'generator.pkl')))
            self.assertTrue(os.path.isfile(os.path.join(f, 'inputs.pkl')))
        self.cleanup(self.output_dir)
    
    def test_seed(self):
        
        # This test will check to see if all parameters are the same,
        # will the generator create the exact same inputs everytime
        
        random_seed = 53421
        
        input_gen = OC21(self.output_dir, 1, 1, 1, 1, max_index=3, bulk_db='../bulk_oxides_20220621.json', 
                         seed=random_seed, limit_bulk_atoms=50, verbose=True, slabgen_inputs=self.slabgen_inputs)
        input_gen.run()
        
        # now test if we can reproduce the files exactly
        output_dir = os.path.join(cwd, 'seed_test')
        input_gen = OC21(output_dir, 1, 1, 1, 1, max_index=3, bulk_db='../bulk_oxides_20220621.json', 
                         seed=random_seed, limit_bulk_atoms=50, verbose=True, 
                         slabgen_inputs=self.slabgen_inputs)
        input_gen.run()
        
        # compare files for the clean slab
        comp = [f for f in glob.glob('seed_test/*')][0].split('/')[-1]
        clean_folder1 = [f for f in glob.glob(os.path.join('seed_test', comp, '*')) if 'clean' in f][0]
        clean_folder2 = [f for f in glob.glob(os.path.join('vasp_test_folders', comp, '*')) if 'clean' in f][0]
        for file1 in glob.glob(os.path.join(clean_folder1, '*')):
            vasp_file = file1.split('/')[-1]
            file2 = os.path.join(clean_folder2, vasp_file)
            self.assertTrue(filecmp.cmp(file1, file2))

        ads_folder1 = [f for f in glob.glob(os.path.join('seed_test', comp, '*')) if 'clean' not in f][0]
        ads_folder2 = [f for f in glob.glob(os.path.join('vasp_test_folders', comp, '*')) if 'clean' not in f][0]
        for file1 in glob.glob(os.path.join(ads_folder1, '*')):
            vasp_file = file1.split('/')[-1]
            file2 = os.path.join(ads_folder2, vasp_file)
            self.assertTrue(filecmp.cmp(file1, file2))
                            
        self.cleanup(self.output_dir)
        self.cleanup(output_dir)
        
    def cleanup(self, folder):
        for f in glob.glob(os.path.join(folder, '*')):
            if 'init' not in f:
                if os.path.isfile(f):
                    os.remove(f)
                else:
                    shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
