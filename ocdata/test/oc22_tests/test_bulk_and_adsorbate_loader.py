import unittest, sys, json, itertools, os
# sys.path.append(os.getcwd().replace(os.getcwd().split('/')[-1], ''))
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.periodic_table import Element
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher

from ocdata.oc22_dataset.adsorbate_configs import adslist, OOH_list

sm = StructureMatcher()


__author__ = "Richard Tran"



class BulkDataSetTest(PymatgenTest):
    def setUp(self):

        # load bulk pmg entries
        self.bulk_dataset = [ComputedStructureEntry.from_dict(d) 
                             for d in json.load(open('../../oc22_dataset/bulk_oxides_20220621.json'))]
                
        # get elemental space
        self.elements_to_do = [el for el in Element 
                               if el.is_metal and el.number < 84 and el.block != 'f']
        self.elements_to_do.extend(['Ce', 'Si', 'Se', 'Sb', 'Ge', 'Te', 'As'])
        
        # Sorts all bulk structure entries into 2 dictionaries (for unary and binary oxides) 
        # by elemental composition where the key represent elements in a compound and the 
        # value is a list of entry objects representing bulks with same types of elements e.g. 
        # {('W', 'Re', 'O'): [WReO, WReO2, WRe2O2, W3Re2O, etc...], ...} 
        # and {('Ni', 'O'): [NiO2, NiO], ...}
        self.comp_dict = {}
        unary_comp_dict = {tuple(sorted([str(el), 'O'])): [] for el in self.elements_to_do}
        binary_comp_dict = {tuple(sorted([str(c[0]), str(c[1]), 'O'])): [] 
                            for c in itertools.combinations(self.elements_to_do, 2)}
        for entry in self.bulk_dataset:
            k = tuple(sorted(entry.composition.as_dict().keys()))
            if len(k) == 2:
                unary_comp_dict[tuple(sorted(k))].append(entry)
            else:
                binary_comp_dict[tuple(sorted(k))].append(entry)
                
        self.comp_dict.update(unary_comp_dict)
        self.comp_dict.update(binary_comp_dict)
        self.binary_comp_dict = binary_comp_dict
        self.unary_comp_dict = unary_comp_dict

    def test_bulk_dataset(self):
        """
        This test will enumerate through all ComputedStructureEntries that we will 
        be investigated in the current iteration of OC21 to determine if each material 
        considered passes the criteria agrees upon in previous discussions:
            - Whether the dataset has at least 5k entries
            - Whether all these entries are symmetrically distinct
            - Whether all entries have fewer than 200 atoms in the convetional unit cell
            - Whether all entries have an energy above hull below 200 meV. We make a few 
                concessions to this last criteria in order to ensure we have as diverse a 
                sample set from the allow chemical space as possible. e.g. if a specific 
                combination of elements does not yield any materiasl with energy above 
                hull = 200 meV, we will add the 7 more stable materials anyways. Furthermore, 
                we want to include all combinations of elements to form artificial rutile 
                structures, which will not be subjec to this. 
        """
        
        # check dataset greater than 5k
        self.assertTrue(len(self.bulk_dataset) > 4000)
        
        no_metastable = []
        # check less than 200 atoms and less than 200 meV.
        self.assertTrue(all([len(entry.structure) < 200 for entry in self.bulk_dataset]))
        self.assertTrue(all([entry.data['e_above_hull'] < 1.5 
                             for entry in self.bulk_dataset if 'e_above_hull' in entry.data.keys()]))
                        
    def test_adsorption_set(self):
        
        self.assertEqual(len(adslist), 9) 
        self.assertEqual(len(OOH_list), 2)
            

if __name__ == "__main__":
    unittest.main()
