
import numpy as np
import pickle

'''
This class handles all things with the adsorbate.
Selects one (either specified or random), and stores info as an object
'''

class Adsorbate():
    def __init__(self, adsorbate_database):
        self.choose_adsorbate_pkl(adsorbate_database)

    def choose_adsorbate_pkl(self, adsorbate_database):
        '''
        Chooses an adsorbate from our pkl based inverted index at random.

        Args:
            adsorbate_database   A string pointing to the a pkl file that contains
                                 an inverted index over different adsorbates.
        Sets:
            atoms                    `ase.Atoms` object of the adsorbate
            smiles                   SMILES-formatted representation of the adsorbate
            bond_indices             list of integers indicating the indices of the atoms in
                                     the adsorbate that are meant to be bonded to the surface
            adsorbate_sampling_str   Enum string specifying the sample, [index]/[total]
        '''
        with open(adsorbate_database, 'rb') as f:
            inv_index = pickle.load(f)
        element = np.random.choice(len(inv_index))

        self.adsorbate_sampling_str = str(element) + "/" + str(len(inv_index))
        self.atoms, self.smiles, self.bond_indices = inv_index[element]
