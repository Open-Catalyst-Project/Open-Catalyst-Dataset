"""
This module enumerates through all possible terminations (O-vacancies) through a brute
    force enumeration of all possible combinations of surface sites in
    order to create slabs of varying coverages and stoichiometries.
    To be moved to pymatgen once its ready.
"""

import numpy as np
import itertools
import random
import warnings

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.core.structure import Structure, StructureError

from ocdata.oc22_dataset.surface import get_symmetrically_distinct_miller_indices, SlabGenerator, center_slab
from ocdata.oc22_dataset.MXide_adsorption import AdsorbateSiteFinder, \
make_superslab_with_partition, label_sites, get_repeat_from_min_lw

__author__ = "Richard Tran"
__version__ = "0.1"
__maintainer__ = "Richard Tran"
__email__ = "rtran@andrew.cmu.edu"
__date__ = "3/10/21"



def get_all_terminations(slabgen, check_polar=False):
    slabs = slabgen.get_slabs(symmetrize=True, check_polar=check_polar)
    return slabs
def get_tasker2(slabgen, repeat):
    slabs = slabgen.get_slabs()
    taskers = []
    for slab in slabs:
        slab.make_supercell(repeat)
        tslabs = slab.get_tasker2_slabs()
        tslabs = [slab for slab in tslabs if not slab.is_polar()]
        # check if any nonpolar slabs have same cations terminating both sides
        for slab in tslabs:
            sites = sorted(slab, key=lambda site: site.c)
            for site1 in sites:
                if '+' in site1.species_string:
                    break
            for site2 in reversed(sites):
                if '+' in site2.species_string:
                    break

            if site1.species_string == site2.species_string:
                taskers.append(slab)
    return taskers
    
def get_half_terminations(slabgen, check_polar=False):
    slabs = slabgen.get_slabs()
    symslabs = []
    for slab in slabs:
        sym = slabgen.nonstoichiometric_symmetrized_slab(slab, both=False, check_polar=check_polar)
        symslabs.extend(sym)
    return symslabs
def get_slab_from_random_shift_both_surfs(slabgen, r, check_polar=False):
    shift = r.sample(slabgen._calculate_possible_shifts(), 1)[0]
    slab = slabgen.get_slab(shift=shift)
    slabs = slabgen.nonstoichiometric_symmetrized_slab(slab, both=True, check_polar=check_polar)
    return slabs
def get_slab_from_random_shift_one_surf(slabgen, r, check_polar=False):
    shift = r.sample(slabgen._calculate_possible_shifts(), 1)[0]
    slab = slabgen.get_slab(shift=shift)
    slabs = slabgen.nonstoichiometric_symmetrized_slab(slab, both=False, check_polar=check_polar)
    return slabs

def get_random_clean_slabs(bulk, n_slabs, max_index, min_slab_size, min_vacuum_size, 
                           lll_reduce=False, in_unit_planes=False, primitive=False, 
                           max_normal_search=None, min_lw=8.0, xsite='O', height=0.9, 
                           r=None, max_ssize=300, check_polar=True):
    '''
    A class that generates a random clean (unadsorbed slab) as a reference for the adsorbed 
        slab. Input will be a bulk structure. We randomly select a facet (miller index) for 
        that bulk structure whose surface we will model. Once that is done, we fully saturate 
        the two surfaces of the slab model with O-lattice positions. This will allow us to 
        sample the full set of combinations of surface oxygen vacancies. This set will 
        incorporate all possible terminations for an oxide surface including: simple cleavage, 
        stoichiometric, non-stoichiometric, random vacancies, and Tasker II surfaces. We 
        randomly sample a coverage of surface O-lattice positions to remove on the surface to 
        create termination and that becomes our slab. Note that these slabs always have 
        symmetrically equivalent surfaces on both sides.

        bulk (pmg Stucture): Initial input structure. Note that to ensure that the miller
            indices correspond to usual crystallographic definitions, you should supply a
            conventional unit cell structure.
        n_slabs (int): N number of random clean slabs
        max_index (int): The maximum index. For example, a max_index of 1 means that
            (100), (110), and (111) are returned for the cubic structure. All other
            indices are equivalent to one of these.
        xsite (Element string): The anion in the bulk, usually O, N, S, C etc
        min_lw (float): Minimum length/width of the slab cross sectional dimensions
        max_ssize (int): Omit slab if it exceeds this number of atoms
        check_polar (bool): Check if slab is polar, will omit if it is
        r (random): Python random function for creating random slab
        height (float): distance from topmost atom from which to consider surface atoms
        
        See pymatgen.core.surface.SlabGenerator for all other parameters
    '''

    random_terminations = []
    bulk = bulk.copy()
    
    # add oxidation states to find nonpolar surface in case symmetry does not work
    if check_polar:
        oxistates = bulk.composition.oxi_state_guesses()
        if not oxistates:
            nM, nO = 0, 0
            comp = bulk.composition.as_dict()
            for el in comp.keys():
                if el != 'O':
                    nM+=comp[el]
                else:
                    nO+=comp[el]
            (nO*2)/nM
            oxistates = [{el: (nO*2)/nM  if el != 'O' else -2 for el in comp.keys()}]
        bulk.add_oxidation_state_by_element(oxistates[0])
    
    miller_list = get_symmetrically_distinct_miller_indices(bulk, max_index)
    r = r if r else random
    r.shuffle(miller_list)

    for hkl in miller_list:
        # randomly select a termination created from simple cleavage
        ssize = 1.5*(bulk.lattice.c) if 2*bulk.lattice.c > min_slab_size else min_slab_size
        slabgen = SlabGenerator(bulk, hkl, ssize, min_vacuum_size, 
                                lll_reduce=lll_reduce, center_slab=True, 
                                in_unit_planes=in_unit_planes, primitive=primitive,
                                max_normal_search=max_normal_search, symprec=0.3)
        
        # your slab needs at least 2xnatoms of your ouc to be valid, if natoms/2 is 
        # already larger then the max alotted slab size, clearly the slab's gonna be too large
        if len(slabgen.oriented_unit_cell) > max_ssize/2:
            continue
        slab = slabgen.get_slab()
        repeat = get_repeat_from_min_lw(slab, min_lw)
        if len(slab)*np.linalg.norm(repeat[0])*np.linalg.norm(repeat[1]) > max_ssize:
            continue
        
        slabs = get_all_terminations(slabgen, check_polar=check_polar)
        if len(slabs) == 0:
            try:
                slabs = get_tasker2(slabgen, repeat)
            except ValueError:
                continue
            if len(slabs) == 0:
                continue
            slab = r.sample(slabs, 1)[0]
            slab.remove_oxidation_states()
            slab.oriented_unit_cell.remove_oxidation_states()
            random_terminations.append(slab)
            pn = len(random_terminations)
            random_terminations = slab_generator_fail_safe(random_terminations)
            if len(random_terminations) == n_slabs:
                break
            else:
                continue
        
        max_Ocontent = max([slab.composition.fractional_composition.as_dict()['O'] for slab in slabs])
        satslabs = [slab for slab in slabs if 
                    slab.composition.fractional_composition.as_dict()['O'] == max_Ocontent]
        slab = random.sample(satslabs, 1)[0]
        
        slab = AdsorbateSiteFinder(slab, height=0.9).slab.copy()
        slab = label_sites(slab)
        slab.make_supercell(repeat)
        
        # skip vacancy generator if our criteria is based on polarity
        if check_polar and not slab.is_symmetric():
            slab.remove_oxidation_states()
            slab.oriented_unit_cell.remove_oxidation_states()
            random_terminations.append(slab)
            pn = len(random_terminations)
            random_terminations = slab_generator_fail_safe(random_terminations)
            if pn != 0 and primitive == True:
                primitive = False
                max_normal_search = None
            if len(random_terminations) == n_slabs:
                break

        # generate a random termination
        top_sites = return_top_sites(slab, species=[xsite])

        # Make a new termination from randomly selected coverage and                         
        # O-latt positions. If there are too many top_sites, the number   
        # of possible combinations explode like crazy, causing python  
        # crash. Lets limit the number of combinations we iterate through
        # O-latt positions. 
        new_slab = slab.copy()
        if len(top_sites) > 0:
            i = r.sample(range(1, len(top_sites)+1), 1)[0]
            c = r.sample(top_sites, i)
            new_slab.symmetrically_remove_atoms(c, tol=0.2)
        
        ####################################################################################
        # need to fix this later...
        if not new_slab.is_symmetric(symprec=0.1):
            continue
        ####################################################################################
        
        new_slab.remove_oxidation_states()
        new_slab.oriented_unit_cell.remove_oxidation_states()
        random_terminations.append(new_slab)
        pn = len(random_terminations)
        random_terminations = slab_generator_fail_safe(random_terminations)
        if pn != len(random_terminations) and primitive == True:
            primitive = False
            max_normal_search = None
        if len(random_terminations) == n_slabs:
            break
            
    return random_terminations

def all_surface_site_combination(slab, coverage_matrix, species=[], verbose=False):
    """
    Must take in a symmetric slab
    """

    slab = slab.copy()
    slab.make_supercell(coverage_matrix)
    top_sites = return_top_sites(slab, species=species)

    # enumerate through all possible combos of top sites and remove
    # combos from top and bottom surfaces to make a new termination
    sm = StructureMatcher()
    all_terms = []
    for i, topsite in enumerate(top_sites):
        sub_terms = []
        for c in itertools.combinations(top_sites, i):
            new_slab = make_vacancies(slab, top_sites, c)
            ####################################################################################
            # need to fix this later...
            if not new_slab.is_symmetric():
                continue
            ####################################################################################
            sub_terms.append(new_slab)
        if verbose:
            print('combos: ', i, len(sub_terms))
        sub_terms = [g[0] for g in sm.group_structures(sub_terms)]
        if verbose:
            print('after grouping: ', i, len(sub_terms))
        all_terms.extend(sub_terms)

    # input slab corresponds to r=n for nCr, i.e. keep all
    all_terms.append(slab.copy())

    return all_terms

def return_top_sites(labelled_slab, species=[]):
    # get all top surface sites of designated species
    top_sites = []
    for i, site in enumerate(labelled_slab):
        if site.site_type == 'Top':
            if species and site.species_string not in species:
                continue
            else:
                top_sites.append(i)
    return top_sites

def make_vacancies(labelled_slab, top_sites, c):

    new_slab = labelled_slab.copy()
    for tsite in top_sites:
        if tsite not in c:
            # need to remove sites one by one because
            # each site removal changes the surface symmetry
            site_i = new_slab.index(labelled_slab[tsite])
            new_slab.symmetrically_remove_atoms([site_i], tol=0.2)

    return new_slab

def slab_generator_fail_safe(slabs, min_c_size=10, percentage_slab=0.2, 
                             percentage_fully_occupied_c=0.8, list_of_slabs=True):
    """
    Slab input analyzer double checks the input slabs are generated correctly. Will sort slabs based on:
        1. Whether or not the c lattice parameter is inappropriately thin (less than min_c_size). Note 
            that min_c_size should be the min_vacuum_size+min_slab_size that was used in SlabGenerator
        2. Whether there are actually enough atoms to make a slab representative of the material. By 
            default, the slab should obviously have more atoms than the oriented unit cell used to 
            build it, otherwise something went wrong...
        3. Whether the slab is too thin which is defined by the pecentage of the lattice along c 
            being occupied by a slab layer (percentage_slab).
        4. Whether the slab's hkl plane has been appropriately oriented parallel to the xy plane of the 
            slab lattice. If the atoms are occupying a pecentage of the the lattice along c defined by 
            percentage_fully_occupied_c, then the slab was probably reoriented along the xz or yz plane 
            which really messes with a lot of the analysis and input generation.
    
    If a slab passes all these check, then it should be fine for the most part. Ideally once these issues 
    have been fixed in the generator, we won't need this anymore...
    
    params::
        slabs (pmg Slab): List of Slab structures to check
        min_c_size (float Å): minimum c lattice parameter you expect your slab model to have
        percentage_slab (float): if the slab layer of the lattice only occupies this percentage
            of the model along the c direction, its way too thin
        percentage_fully_occupied_c (float): if the slab layer of the lattice occupies this 
            percentage of the model along the c direction, it probably means the hkl plane was 
            reoriented parallel to xz or yz.
    
    Returns list of slabs (default) or list of errors
    """
    
    slab_correct, slab_errors = [], []
    for slab in slabs:
        # note please change min_c_size to whatever the actual minimum 
        # c-lattice parameter should be when anaylzing the actual set
        if slab.lattice.c < min_c_size: 
            slab_errors.append('bad_c_lattice')
            continue
        elif len(slab.oriented_unit_cell) > len(slab):
            slab_errors.append('not_enough_atoms')
            continue

        if any([site.frac_coords[2] > 0.9 for site in slab]) or \
        any([site.frac_coords[2] < 0.1 for site in slab]):
            slab = center_slab(slab)

        top_atoms, bottom_atoms  = [], []
        ccords = [site.coords[2] for site in slab]
        if (max(ccords) - min(ccords))/slab.lattice.c < percentage_slab:
            slab_errors.append('too_thin')
            continue

        ccoords = []
        for site in slab:
            ccoords.append(site.coords[2])
        if (max(ccoords)-min(ccoords))/slab.lattice.c > percentage_fully_occupied_c:
            slab_errors.append('xy_not_hkl')
            continue
        
        try:
            s = Structure(slab.lattice, slab.species, slab.frac_coords, validate_proximity=True)
            
        except StructureError:
            slab_errors.append('validate_proximity_error')
            continue
        
        slab_correct.append(slab)
        
    if list_of_slabs:
        if not slab_correct:
            print(slab_errors)
        return slab_correct
    return slab_errors


"""
Unittests and warnings

ToDo:
-Whats the difference between this and the T2 generator? Its more
    generalized because it enumerates through all possible
    coverages by decreasing the number of sites it can play with
    for each iteration of the main for loop

Unittests to implement:
    -Check if site labelling captures 1-to-1 surface sites on both surfaces
    -Does this cover all possible Tasker combinations as well
        (including stoichiometric Tasker 2)?
    -Why are the number of distinct terminations not the same if I
        use an OUC of slightly different latt params?

Warnings:
    -Slab not symmetric
    -Cannot do exact 1-to-1 mapping of surface sites
"""