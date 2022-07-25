import numpy as np
from scipy.spatial import Delaunay
import itertools, copy, random, time

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import *
from pymatgen.util.coord import in_coord_list_pbc, pbc_shortest_vectors, \
all_distances, lattice_points_in_supercell, coord_list_mapping_pbc
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ocdata.oc22_dataset.surface import * 



class MXideAdsorbateGenerator(AdsorbateSiteFinder):
    """
    Adapted as a child class of AdsorbateSiteFinder in pymatgen, but with new methods for finding
        adsorption sites based on surface lattice positions of metal-Xides (X= O, N, C, etc). 
        Includes functionality for coverage and partial rotational degree of freedom. Works by 
        first enumerating through all possible adsorbate positions (see MX_adsites() and mvk_adsites() 
        for more details). Then the user has two options for adslab generation:
        1. generate_all_MXide_adsorption_structures(): Will enumerate all possible rotational
            DoF, coverages and combinations of MX_adsites and mvk_adsites. Considering the number
            of variables going into finding these sites, its not reccomended that you use this
            function or else you will be generating hundreds of thousands of adsorbate
            configurations. Further analysis and work needs to be implemented to reduce the
            number of possible configurations one would enumerate through
        2. generate_random_MXide_adsorption_structure(): Of all the possible combinations of
            coverage, rotation and xyz adsorbate positions, we sample an N number of configs
            to create adslabs out of.

    .. attribute:: init_slab 
    
        The intiial slab provided to MXideAdsorbateGenerator
        
    .. attribute:: slab
        
        The initial slab is then expanded in its cross sectional dimensions 
            with a minimum length of min_lw. Each section of this superslab
            will be labelled in partitions, e.g. a 2x2 slab of the initial 
            primitive slab will be labelled with 4 partitions. This will 
            help with identifying and reducing symmetrically equivalent adslabs
            with more than one adsorbate. Will also label the super slab with
            the bulk Wyckoff positions and the surface vs bulk sites
    
    .. attribute:: X

        Non-metal element of the compound (typically O, N, S, C, etc)
                
    .. attribute:: bulk 
        
        The oriented unit cell of the slab

    .. attribute:: bondlength

        M-O bond length of a bulk structure.
        
    .. attribute:: surf_metal_sites 
        
        List of surface cations    

    .. attribute:: MX_adsites

        Adsorbate positions on the surface. If the molecule contains X, the X site
            will full the position of the missing X lattice position on the surface
            
    .. attribute:: MX_partitions
    
        Returns the corresponding labels for the partitions 
            corresponding to the MX_adsites list

    .. attribute:: mvk_adsites

        Alternatively, we can also find adsorbate positions on the surface by assuming
            the adsorbate will want to form a new molecule with an X lattice position
            existing on the surface.
            
    .. attribute:: mvk_partitions
    
        Returns the corresponding labels for the partitions 
            corresponding to the mvk_adsites list
            
    .. attribute:: bulk_wyckoff_cn
    
        Dictionary of each Wyckoff symbol and their coordination number, used 
            for broken bond analysis
            
    .. attribute:: min_adsorbate_dist
    
        Configurations with multiple adsorbates within this distance of each otehr are omitted
        
    .. attribute:: sm
    
        StructureMatcher() from pymatgen
                
    """

    def __init__(self, slab, selective_dynamics=False,
                 height=1.25, mi_vec=None, repeat=None, min_lw=8.0, 
                 verbose=True, max_r=6, tol=1.1, adsorb_both_sides=False,
                 ads_dist_is_blength=True, ads_dist=False, r=None, relax_tol=1e-5, 
                 positions=['MX_adsites', 'mvk_adsites']):

        """
        slab (pmg Slab): slab object for which to find adsorbate sites
        selective_dynamics (bool): flag for whether to assign
            non-surface sites as fixed for selective dynamics
        height (float): height criteria for selection of surface sites
        mi_vec (3-D array-like): vector corresponding to the vector
            concurrent with the miller index, this enables use with
            slabs that have been reoriented, but the miller vector
            must be supplied manually
        repeat (3-tuple): repeat for getting extended surface mesh
        min_lw (float): minimum length and width of the slab, only used
            if repeat is None
        verbose (bool): Bool to tell code to print out additional 
            information about slab generation
        max_r (float): Maximum radius for locating the M-X bondlength in a bulk
        tol (float): tol value for finding bondlength
        adsorb_both_sides (bools): Bool whether to adsorb booth 
            surfaces. Slab symmetry is maintained when doing so
        ads_dist_is_blength (bool): The minimum distance allowed between two 
            adsorbates on the surface is the distance of the M-X bondlength
        ads_dist (float): User selected minimum distance allowed between two 
            adsorbates on the surface
        r (random): Python random function with pre-specified seed number 
        relax_tol (float): Accounts for minor relaxations of surface atoms in
            order to identify adsorption sites
        positions(['MX_adsites', 'mvk_adsites']): Determines the type of adsorbate 
            sites to consider when constructing adsorbed slabs. 'MX_adsites' means 
            bind to an undercoordinate surface metal at the lattice position of a 
            vacant atom X. 'mvk_adsites' means bind to existing X atom on the surface 
            in such a way that you form something like a new molcule 
        """

        self.init_slab = slab.copy()
        super().__init__(
            self.init_slab, selective_dynamics=selective_dynamics,
            height=height, mi_vec=mi_vec)        

        # Create super slab cell from unit slab
        if repeat is None:
            repeat = get_repeat_from_min_lw(self.slab, min_lw)
        self.slab = make_superslab_with_partition(self.slab, repeat)
        self.slab = label_sites(self.slab)

        # Works with structures containing only 1 X atom for now, i.e. oxides
        # nitrides, sulfides etc. Will generalize for all systems later.
        start = time.time()
        self.X = [el for el in self.init_slab.composition.as_dict().keys() 
                  if not Element(el).is_metal and not Element(el).is_metalloid and el != 'Se'][0]
        self.verbose = verbose

        self.relax_tol = relax_tol
        self.bulk = slab.oriented_unit_cell.copy()

        self.bondlength = self.get_bond_length(max_r=max_r, tol=tol)

        # Get surface metal sites 
        self.surf_metal_sites = [site.coords for site in self.slab if site.surface_properties == 'surface' and 
                                 site.species_string != self.X and site.frac_coords[2] > self.slab.center_of_mass[2]]

        # Get bulk-like adsorption sites on top surface where adsorbate binds to M site
        if 'MX_adsites' in positions:
            self.MX_adsites, self.MX_partitions = self.get_bulk_like_adsites()
        else:
            self.MX_adsites, self.MX_partitions = [], []
            
        # Get bulk-like adsorption sites on top surface where
        # adsorbate binds to X site to form the desired molecule
        if 'mvk_adsites' in positions:
            self.mvk_adsites, self.mvk_partitions = self.get_surface_Xsites()
        else:
            self.mvk_adsites, self.mvk_partitions = [], []
        end = time.time() 

        if self.verbose:
            print('Found total adsites: %s for slab of %s atoms in %.3f secs' \
                  %(len(self.MX_adsites) + len(self.mvk_adsites), len(self.slab), end-start))

        # Get CN of bulk Wyckoff position for BB analysis at surface later on
        bulk_wyckoff_cn = {}
        eq = self.bulk.site_properties['bulk_equivalent']
        for i, w in enumerate(self.bulk.site_properties['bulk_wyckoff']):
            sym = str(w) + str(eq[i])
            if sym not in bulk_wyckoff_cn.keys():
                site = self.bulk[i]
                if site.species_string == self.X:
                    bulk_wyckoff_cn[sym] = len([nn for nn in self.bulk.get_neighbors(site, self.bondlength)
                                                if nn.species_string != self.X])
                else:
                    bulk_wyckoff_cn[sym] = len([nn for nn in self.bulk.get_neighbors(site, self.bondlength)
                                                if nn.species_string == self.X])
        self.bulk_wyckoff_cn = bulk_wyckoff_cn
        if ads_dist:
            self.min_adsorbate_dist = ads_dist
        elif ads_dist_is_blength:
            self.min_adsorbate_dist = self.bondlength
        else:
            self.min_adsorbate_dist = self.calculate_min_adsorbate_dist()
        self.sm = StructureMatcher()
        self.adsorb_both_sides = adsorb_both_sides
        self.random = r if r else random    
            
            
    def get_bond_length(self, max_r=6, tol=1.1):
        """
        Get the nearest neighbor bond length under a maximum radius (max_r) 
        between metal and oxygen (with a (tol-1)x100% tolerance for sites 
        close enough to be neighbors)
        Args:
            max_r (float in Å): Max radius from nonmetal site to look for neighbors, defaults to 6Å
            tol (float): Percentage tolerance of determined bond length to account for numerical
            errors or neighboring sites slightly further away
            
        Returns: 
            M-O bond length
        """
        
        min_dists = []
        for site in self.bulk:
            if site.species_string != self.X:
                ndists = [nn.distance(site) for nn in self.bulk.get_neighbors(site, max_r) 
                             if nn.species_string == self.X]
                ndists = [2] if not ndists else ndists
                min_dists.append(min(ndists))
        return min(min_dists) * tol
    
    ########################## NEED A BETTER ALGO FOR ADSORBATE DISTANCES ##########################
    
    def calculate_min_adsorbate_dist(self, tol=0.05):
        """
        Set minimum adsorbate distance for multiple adsorbate coverages. Lets make it so
        that the minimum distance is the shortest distance of two surface oxygen sites.
        If the surface is O-defficient, make the minimum distance the shortest distance
        between two missing lattice positions (essentially the same thing)
        Args:
            tol (float): Percentage tolerance of determined minimum distance to
             account for numerical errors or neighboring sites slightly further away
        
        Returns:
            Minimum distance allowed between adsorbates
        """
        
        # First we tag all surface oxygen or surface oxygen vacancies (self.MX_adsites)
        min_dist = []
        pseudo_slab = self.slab.copy()
        if self.MX_adsites:
            for coord in self.MX_adsites:
                pseudo_slab.append(self.X, coord, coords_are_cartesian=True,
                                   properties={"surface_properties": 'pseudo'})
        else:
            props = []
            for p in pseudo_slab.site_properties['surface_properties']:
                p = 'pseudo' if p == 'surface' else p
                props.append(p)
            pseudo_slab.add_site_property('surface_properties', props)

        # Get the nearest neighbor distances for each of these oxygen and vacancy 
        # sites. The shortest one is the minimum disance allowed between adsorbates
        for site in pseudo_slab:
            if site.surface_properties == 'pseudo' and site.species_string == self.X \
                    and site.frac_coords[2] > self.slab.center_of_mass[2]:
                if len([nn for nn in pseudo_slab.get_neighbors(site, self.bondlength)
                        if nn.species_string != self.X]) == max(self.bulk_wyckoff_cn.keys()):
                    continue

                dists = []
                for nn in pseudo_slab.get_neighbors(site, 5, include_image=True):
                    d = np.linalg.norm(nn.coords - site.coords)
                    if nn.species_string != self.X:
                        continue
                    if d < 1e-5:
                        continue
                    dists.append(d)

                min_dist.append(min(dists))

        return min(min_dist) * (1 - tol)

    ########################## NEED A BETTER ALGO FOR ADSORBATE DISTANCES ##########################

    def find_surface_sites_by_height(self, slab, height=1.25, xy_tol=0.05, both_surfs=False):
        """
        This method finds surface sites by determining which sites are within
        a threshold value in height from the topmost site in a list of sites

        Args:
            slab (Slab): Slab structure to find surface sites on
            height (float): threshold in angstroms of distance from topmost
                site in slab along the slab c-vector to include in surface
                site determination
            xy_tol (float): if supplied, will remove any sites which are
                within a certain distance in the miller plane.
            both_surfs (bool): if True, returns surface sites of both surfaces

        Returns:
            list of sites selected to be within a threshold of the highest
        """

        # Get projection of coordinates along the miller index
        m_projs = np.array([np.dot(site.coords, self.mvec) for site in slab.sites])

        # Mask based on window threshold along the miller index.
        mask = []
        for mproj in m_projs:
            if mproj - np.amax(m_projs) >= -height:
                mask.append(True)
            elif both_surfs and mproj - np.amin(m_projs) <= height:
                # Get bottom surface sites too
                mask.append(True)
            else:
                mask.append(False)

        surf_sites = [slab.sites[n] for n in np.where(mask)[0]]
        if xy_tol:
            # sort surface sites by height
            surf_sites = [s for (h, s) in zip(m_projs[mask], surf_sites)]
            surf_sites.reverse()
            unique_sites, unique_perp_fracs = [], []
            for site in surf_sites:
                this_perp = site.coords - np.dot(site.coords, self.mvec)
                this_perp_frac = slab.lattice.get_fractional_coords(this_perp)
                if not in_coord_list_pbc(unique_perp_fracs, this_perp_frac):
                    unique_sites.append(site)
                    unique_perp_fracs.append(this_perp_frac)
            surf_sites = unique_sites

        return surf_sites

    def assign_site_properties(self, slab, height=1.25, both_surfs=True):
        """
        Assigns site properties. See find_surface_sites_by_height() 
            for parameters description
        """
        if "surface_properties" in slab.site_properties.keys():
            return slab

        surf_sites = self.find_surface_sites_by_height(slab, height, both_surfs=both_surfs)
        surf_props = ["surface" if site in surf_sites else "subsurface" for site in slab.sites]
        slab = slab.copy(site_properties={"surface_properties": surf_props})
        
        return slab

    @classmethod
    def assign_selective_dynamics(cls, slab):
        """
        Helper function to assign selective dynamics site_properties
        based on surface, subsurface site properties

        Args:
            slab (Slab): slab for which to assign selective dynamics
        """
        sd_list = []
        sd_list = [[False, False, False] if site.properties['surface_properties'] == 'subsurface' \
                                            or site.frac_coords[2] < slab.center_of_mass[0]
                   else [True, True, True] for site in slab.sites]
        new_sp = slab.site_properties
        new_sp['selective_dynamics'] = sd_list
        return slab.copy(site_properties=new_sp)

    def get_transformed_molecule_MXides(self, molecule, angles_list, axis=[0, 0, 1]):

        """
        Translates the molecule in such a way that the X atom in the molecules
            sites in the same position as the X atom on the surface. Allows for
            incremental rotation about an axis of rotation. e.g. n_rotations=4,
            and axis=[0,0,1] will create set of 4 more molecules each rotated
            about z-axis by 90, 180, 270, 0 degrees. n_rotations=1, does nothing.

        Args:
            molecule (Molecule or list): molecule (or pair of molecule objects)
                representing the adsorbate (or adsorbate in the first object and
                molecule it forms with the existing lattice position in the second
                object). In the case of a list, the second molecule can be a molecule
                the adsorbate can bind to the X lattice position to create (e.g. [OH, OOH]
                means OH binds to O lattice position on the surface to form OOH), or
                the second molecule can describe the bond one of the atoms in the adsorbate
                will form with the existing surface lattice position (e.g. [HCl, OH]
                means the H in HCl binds to a surface lattice O site to form OH)
            angles_list (list): List of angles to rotate the molecule about the axis
            axis (vector [x,y,z]): Axis in which to rotate molecule about
        """

        translated_molecules = []
        # Enumerate through all n_rotations about axis
        if type(molecule).__name__ == 'list':
            ads, true_ads = molecule[1], molecule[0]
        else:
            ads = molecule

        # For which ever species in the adsorbate whose combination of
        # electronegativity and atomic radius is closest to X site, we
        # bind the adsorbate to the lattice X-vac based on that species
        Xel, norm_attr = Element(self.X), []
        for site in ads:
            el = Element(site.species_string)
            elpt = np.array([el.X / Xel.X, el.atomic_radius / Xel.atomic_radius])
            norm_attr.append(np.linalg.norm(elpt - np.array([1, 1])))
        binding_species = ads[norm_attr.index(min(norm_attr))].species_string
        
        for deg in angles_list:

            # Translate the molecule so that atom X is always at (0, 0, 0).
            # Can return multiple translations if there is more than one
            # species of atom X in the molecule.

            for site in ads:

                # For adsorbates using mvk-like adsorption, assign the template molecule
                # you wish to form on the surface with 'anchor' site_property with the atom
                # substituting the surface atom True while all others are False
                if 'anchor' in ads.site_properties.keys() and not site.anchor:
                    continue

                # Site of the adsorbate whose electronegativity and atomic
                # radius is closest to X attaches to the lattice X-vac
                if site.species_string == binding_species:
                    translated_molecule = ads.copy()
                    translated_molecule.translate_sites(vector=-1 * site.coords)

                    # Substitute the template adsorbate's anchor into the lattice position
                    if 'anchor' in ads.site_properties.keys():
                        translated_molecule.remove_sites([i for i, site in enumerate(translated_molecule) \
                                                          if site.anchor])
                        translated_molecule.remove_site_property('anchor')

                        # If the entire actual adsorbate does not make up a part of the
                        # template adsorbate, there is an additional step where we now
                        # bind the actual adsorbate to the adsorbed template.
                        if not all([el in ads.species for el in true_ads.species]):
                            # Identify the site bonded to the lattice position
                            # and substitute that site with the true adsorbate
                            bonded_site = translated_molecule[0]
                            anchor_index, anchor_site = [[i, site] for i, site in enumerate(true_ads) if \
                                                         site.species_string == bonded_site.species_string][0]
                            translated_molecule = true_ads.copy()
                            translated_molecule.translate_sites(vector=bonded_site.coords - anchor_site.coords)

                    translated_molecule = self.add_adsorbate_properties(translated_molecule)
                    translated_molecule.rotate_sites(theta=deg, axis=axis)
                    setattr(translated_molecule, 'deg', round(deg, ndigits=2))
                    translated_molecules.append(translated_molecule)

        return translated_molecules

    def add_adsorbate_properties(self, ads):
        if "selective_dynamics" in self.slab.site_properties.keys():
            ads.add_site_property("selective_dynamics", [[True, True, True]] * ads.num_sites)
        if "surface_properties" in self.slab.site_properties.keys():
            surfprops = ["adsorbate"] * ads.num_sites
            ads.add_site_property("surface_properties", surfprops)

        return ads

    def get_bulk_like_adsites(self):

        """
        Algorithm:

        Method to get bulk-like X-adsorbate positions in a MXide. Takes in
            slabs with Laue PG symmetry only. First we locate the equivalent
            M-sites on the bottom with the same orientation of its polyhedron
            as the top M-site minus the undercoordinated environment. The X
            atom at the bottom that is missing from the top is then translated
            to the top via a vector betweeen the top and equivalent bottom M-site.
            This is our bulk-like adsorbate position.

            NOTE: Future implementation should use a general bond_dict instead of X
                and bondlength in case of ternary compositions made of more than one X

        Returns:
            (List) of bulk-like adsorption sites (cartesian) for X on the top surface
        """

        com = self.slab.center_of_mass
        adsites, partitions = [], []
        # get all the anions (oxygen)
        surfX = [site.frac_coords for site in self.slab if site.species_string == self.X 
                 and site.surface_properties == 'surface']

        for surfsite in self.slab:
            # consider cations (metal) in slab and check whether or not the metal site is
            # undercoordinated by comparing it to the corresponding site in a bulk structure 
            # (using the Wyckoff symbol). If it is, there are Ovac we can treat as adsorption sites
            if surfsite.species_string != self.X and surfsite.frac_coords[2] > com[2]:
                surf_nn = self.slab.get_neighbors(surfsite, self.bondlength)
                for bulksite in self.bulk:
                    if bulksite.bulk_wyckoff == surfsite.bulk_wyckoff and \
                                    bulksite.species_string == surfsite.species_string:
                
                        cn = len(self.bulk.get_neighbors(bulksite, self.bondlength))
                        break
                if len(surf_nn) == cn:
                    continue
                
                # If the current metal site we are considering is undercoordinated, now we 
                # find a corresponding site of the same bulk Wyckoff in the slab that is fully
                # coordinated and superimpose the missing X positions on the undercoordinated
                # cation in order to identify the positions of the anion vacancy
                for site in self.slab:

                    bulk_frac_coords = [nn.frac_coords for nn in self.slab.get_neighbors(site, self.bondlength)]
                    if len(bulk_frac_coords) == cn:
                        translate = surfsite.frac_coords - site.frac_coords
                        # check that all the positions of the vacancies for our undercoordinated  
                        # site match the positions of the fully coordinated site when translated
                        if all([in_coord_list_pbc(bulk_frac_coords,
                                                  nn.frac_coords - translate, 
                                                  atol=self.relax_tol) for nn in surf_nn]):
                            
                            for fcoords in bulk_frac_coords:
                                transcoord = fcoords + translate
                                # make sure the proposed adsite is not already considered, is on 
                                # the top surface, and does not already occupy any existing position
                                if not in_coord_list_pbc(adsites, transcoord, atol=self.relax_tol) and \
                                                fcoords[2] > site.frac_coords[2] and \
                                        not in_coord_list_pbc(surfX, transcoord, atol=self.relax_tol):
                                    adsites.append(transcoord)
                                    partitions.append(surfsite.supercell)
                            break

        return [self.slab.lattice.get_cartesian_coords(fcoord) for fcoord in adsites], partitions

    def get_surface_Xsites(self):

        # Identify surface sites
        com = self.slab.center_of_mass[2]

        adsites, partitions = [], []
        for i, site in enumerate(self.slab):
            # Find equivalent sites for top surface M-site only
            if site.frac_coords[2] > com and \
                            site.surface_properties == 'surface' and site.species_string == self.X:
                # Get the coordinate X-sites on the top too so we
                # can map to an equivalent environment at the bottom
                adsites.append(site.coords)
                partitions.append(site.supercell)

        return adsites, partitions
    
    def unit_normal_vector(self, pt1, pt2, pt3):
        """
        From three positions on the surface, calculate the vector normal to the plane 
            of those positions. This will make it easier to position specific adsorbates 
            parallel to specific subfacets of the surface like in steps and terraces, 
            similar to the algorithm proposed in CatKit. pt1, pt2, pt3 are three points 
            that define a plane which we calculate the normal vector for
        """
        
        # These two vectors are in the plane
        v1 = np.round(pt3 - pt1, 3)
        v2 = np.round(pt2 - pt1, 3)

        # the cross product is a vector normal to the plane
        vector = np.round(np.cross(v1, v2), 3)
        if np.linalg.norm(vector) == 0 or vector[2] == 0:
            # skip (anti)parallel vectors
            return np.array([False, False, False])
        
        else:
            return vector/np.linalg.norm(vector)
    
    def get_tri_mesh_normal_vectors(self, surface_sites):
        """
        Create a Delaunay triangle mesh on the surface. Then obtain the unit 
        normal vector for each triangle out of the slab. This will allow us 
        to position an adsorbate parallel to any place on the surface. This 
        method of adsorbate placement is inspired and adapted from the 
        following works::

            Montoya, J. H., & Persson, K. A. (2017). "A high-throughput framework 
            for determining adsorption energies on solid surfaces", Npj Computational 
            Materials, 3(1), 14. doi:10.1038/s41524-017-0017-z
        
        as well as::
            
            Boes, J. R., Mamun, O., Winther, K., & Bligaard, T. (2019). "Graph Theory 
            Approach to High-Throughput Surface Adsorption Structure Generation", Journal 
            of Physical Chemistry A, 123(11), 2281–2285. doi:10.1021/acs.jpca.9b00311
        """
    
        # really stupid fix for handling Qhull error, need a cleaner solution
        surface_sites = [coord+np.array([self.random.sample(range(0,1000), 1)[0]])*10**(-9) \
                         for coord in surface_sites]

        # Get a triangular mesh of all the surface sites
        dt = Delaunay(surface_sites)
        triangles = [tri for v in dt.simplices for tri in itertools.combinations(v, 3)]
        triangles = list(set(map(lambda i: tuple(sorted(i)), triangles)))

        # Get a corresponding normal unit vector for every triangle
        triangle_and_norm_dict = {}
        for tri in triangles:
            pt1, pt2, pt3 = [surface_sites[i] for i in tri]
            v = self.unit_normal_vector(pt1, pt2, pt3)
            if not any(v):
                continue
            unorm = v/np.linalg.norm(v)
            unorm = -1*unorm if unorm[2] < 0 else unorm # ensure vector points out of slab
            triangle_and_norm_dict[tuple(unorm)] = [pt1, pt2, pt3]
    
        return triangle_and_norm_dict
    
    def get_dimer_coupling_sites(self, molecule):
        """
        This function locates the coordinate pairs of all dimer coupled monatomic 
            adsorption by geometrically solving for the dimer positions as the two
            points of the top base for a trapezoid using the two metal sites that 
            the dimer binds to as the bottom base of the trapezoid. It returns A 
            list containing all coordinate pairs. The number of dimer coupling pairs 
            that exist are equal to the number of surface M-M pairs with the shortest distance 
        
        arg:
            molecule (Molecule): A pmg Molecule object represent a dimer of two X atoms
            
        Returns:
            The cordinates of the two dimer atoms in the slab cell
        """
        
        OO_length = molecule.get_distance(0, 1)
        slab = self.slab.copy()
        
        # get metal sites w/ pbc
        pbcs = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)]
        metal_pos = [slab.lattice.get_cartesian_coords(slab.lattice.get_fractional_coords(coords)+np.array(trans))
                     for coords in self.surf_metal_sites for trans in pbcs]
        metal_pos.extend(self.surf_metal_sites)

        all_dists = all_distances(metal_pos, metal_pos)
        min_dist = min([min([d for d in dists if d !=0]) for dists in all_dists])
        
        # now locate all delaunay triangles using the surface metal sites and get 
        # the corresponding normal vector for the plane formed by each triangle
        triangle_and_norm_dict = self.get_tri_mesh_normal_vectors(metal_pos)

        # Associate every distinct ray in the triangulation mesh with the  
        # corresponding unit vector. Note that rays have more than one normal 
        # vector associated with it since more than one triangle can share the same ray
        pts_and_norm_v = []
        for unorm in triangle_and_norm_dict.keys():
            tri = triangle_and_norm_dict[unorm]
            for v in itertools.combinations(tri, 2):
                pts_and_norm_v.append([v, np.array(unorm)])

        # Now for each ray with length equivalent to the shortest 
        # surface M-M bond, locate the midpoint of the dimer position
        b = OO_length/2
        dimer_pair_coords = []
        for ii, pt_and_norm in enumerate(pts_and_norm_v):
            pt1, pt2 = pt_and_norm[0]
            if "%.2f" %(np.linalg.norm(pt1-pt2)) !="%.2f" %(min_dist):
                continue
            
            # now using the MM coord pairs as the base of our trapezoid, 
            # find the height of the trapezoid represented as a vector.
            tri_base = (min_dist - OO_length)/2 # half base of isosceles triangle
            h = (self.bondlength**2 - tri_base**2)**(1/2) # solve for h
            
            # using the height, MM midpoint, the unit normal vector and 
            # dimer length, solve for the two positions of the dimer coupling
            midpt = (pt1+pt2)/2 
            MM_unit_v = (pt1-pt2)/np.linalg.norm(pt1-pt2)
            unit_v = pt_and_norm[1]
            coord1 = -1*b*MM_unit_v+midpt + h*unit_v
            coord2 = b*MM_unit_v+midpt + h*unit_v
            dimer_midpt = midpt+h*unit_v
            dimer_pair_coords.append([coord1, coord2])

        return dimer_pair_coords

    def is_pos_too_close(self, coords1, coords2):
        """
        skip if the position of two adsorbate species too close
        """

        fcoord1 = self.slab.lattice.get_fractional_coords(coords1)
        fcoord2 = self.slab.lattice.get_fractional_coords(coords2)
        d = min([np.linalg.norm(v) for v in pbc_shortest_vectors(self.slab.lattice, fcoord1, fcoord2)])
        
        return d < self.min_adsorbate_dist
    
    def are_any_ads_too_close(self, adslab, adsites_indices):
        """
        When decorating surface with multiple adsorbates, make sure
        all adsites are at a minimum distance away from each other
        """

        socially_distanced = []
        fcoords1 = [adslab[i].frac_coords for i in adsites_indices if adslab[i].frac_coords[2] > adslab.center_of_mass[2]]
        fcoords2_ads = [site.frac_coords for ii, site in enumerate(adslab) if site.surface_properties == 'adsorbate' 
                    and ii not in adsites_indices and adslab[ii].frac_coords[2] > adslab.center_of_mass[2]]
        fcoords2_surf = [site.frac_coords for ii, site in enumerate(adslab) if site.surface_properties == 'surface' 
                         and ii not in adsites_indices and adslab[ii].frac_coords[2] > adslab.center_of_mass[2]]
        if not fcoords2_ads and not fcoords2_surf:
            return False
        alldists, alldists_surf = [], []
        for vectors in pbc_shortest_vectors(adslab.lattice, fcoords1, fcoords2_ads):
            alldists.append(min([np.linalg.norm(v) for v in vectors]))
        for vectors in pbc_shortest_vectors(adslab.lattice, fcoords1, fcoords2_surf):
            alldists_surf.append(min([np.linalg.norm(v) for v in vectors]))
        return any([min(alldists) < self.min_adsorbate_dist, min(alldists_surf) < 1])
    
    def get_all_rot_and_pos_combs(self, molecule, angles_list, axis=[0, 0, 1], as_list=False):
        """
        Algorithm for enumerating through all possible combinations of molecule 
            rotation + positions. See get_transformed_molecule_MXides() for 
            description of parameters. For dimer coupling, there is now rotational 
            degree of freedom as the positions are predetermined by the positions 
            of adjacent M sites. First we identify all possible positions of 
            adsorbates on the metal. If the Molecule provided is a list of 2 molecules
            with the second being the possible result from binding with oxygen, then we 
            also locate all adsorbate positions for binding to surface oxygen. 
                
        Returns:
            We can either return a list of lists of all combinations. The nested list 
            has the rotated molecule, the position of the adsorbate and the partition 
            which this adsorabte exists in. Can also return a dict where the key is 
            the coordinate of the adsorbate
        """
        
        # separate algorithm returns dimer coupled adsorbates as is
        if hasattr(molecule, 'dimer_coupling'):
            rot_and_pos_combs = []
            coords = self.get_dimer_coupling_sites(molecule)
            # Translate the molecule so that atom X is always at (0, 0, 0).
            for coord_pair in coords:
                if str(coord_pair[0][0]) == 'nan':
                    continue
                transformed_molecule = Molecule(molecule.species, coord_pair)
                transformed_molecule = self.add_adsorbate_properties(transformed_molecule)
                transformed_molecule.translate_sites(vector=-1*coord_pair[0])
                rot_and_pos_combs.append([transformed_molecule, coord_pair[0], float('nan')])
            return rot_and_pos_combs

        # returns both metal adsites and oxygen adsites if the Molecule is a list of molecules
        if type(molecule).__name__ == 'list':
            bulk_like_trans_molecules = self.get_transformed_molecule_MXides(molecule[0], axis=axis,
                                                                             angles_list=angles_list)
            mvk_like_trans_molecules = self.get_transformed_molecule_MXides(molecule, axis=axis,
                                                                            angles_list=angles_list)
        else:
            # Otherwise we just return the metal adsorbate sites
            bulk_like_trans_molecules = self.get_transformed_molecule_MXides(molecule, axis=axis,
                                                                             angles_list=angles_list)
            mvk_like_trans_molecules = []

        if as_list:
            # Get all possible combinations of rotated molecules and adsites
            rot_and_pos_combs = [[mol, site, self.MX_partitions[i]] for i, site in enumerate(self.MX_adsites) 
                                 for mol in bulk_like_trans_molecules]
            rot_and_pos_combs.extend([[mol, site, self.mvk_partitions[i]] for i, site in enumerate(self.mvk_adsites) 
                                      for mol in mvk_like_trans_molecules])
            return rot_and_pos_combs
        
        # Create a dictionary with hte key as the adsorbate coordinate and return that instead
        rot_and_pos_combs = {}
        for mol in bulk_like_trans_molecules:
            for i, site in enumerate(self.MX_adsites):
                if tuple(site) not in rot_and_pos_combs.keys():
                    rot_and_pos_combs[tuple(site)] = []
                rot_and_pos_combs[tuple(site)].append([mol, site, self.MX_partitions[i]])
        for mol in mvk_like_trans_molecules:
            for i, site in enumerate(self.mvk_adsites):
                if tuple(site) not in rot_and_pos_combs.keys():
                    rot_and_pos_combs[tuple(site)] = []
                rot_and_pos_combs[tuple(site)].append([mol, site, self.mvk_partitions[i]])

        return rot_and_pos_combs
    
    def adslab_from_adsites(self, adsite_set, coverage):

        adslab = self.slab.copy()
        adsorbates_too_close = False

        # sorted by molecule
        sorted_adsite_set = {}
        for site in adsite_set:
            c = tuple(site.ads_coord)
            if c not in sorted_adsite_set.keys():
                sorted_adsite_set[c] = []
            sorted_adsite_set[c].append(site)
        
        invalid = False
        indices_to_remove = []
        for ads_coord, molecule in sorted_adsite_set.items():
            if self.adsorb_both_sides:
                mol_props = {}
                for d in [m.properties for m in molecule]:
                    for k, v in d.items(): 
                        mol_props.setdefault(k, []).append(v)                
                adslab.symmetrically_add_atoms([site.species_string for site in molecule], 
                                               [site.coords for site in molecule], 
                                               coords_are_cartesian=True, properties=mol_props)
            else:
                adslab_copy = adslab.copy()
                try:
                    for site in molecule:
                        adslab_copy.append(site.specie, site.coords, coords_are_cartesian=True,
                                      properties=site.properties, validate_proximity=True)
                except ValueError:
                    invalid = True
                    continue

                for site in molecule:
                    adslab.append(site.specie, site.coords,
                                  coords_are_cartesian=True,
                                  properties=site.properties)

            if coverage not in [1, len(self.MX_adsites)+len(self.mvk_adsites)]:
                # check once more if any atoms in adsorbed molecule is too close
                nmol = len(molecule)*2 if self.adsorb_both_sides else len(molecule)
                adsites_indices = [len(adslab) - i for i in range(1, nmol + 1)]
                #################### fcoords are not matching up when using pbc_shortest_vectors ####################
                try:
                    if self.are_any_ads_too_close(adslab, adsites_indices):
                        indices_to_remove.extend(adsites_indices)
                except ValueError:
                    indices_to_remove.extend(adsites_indices)
                    
        adslab.remove_sites(indices_to_remove)
        if not [site for site in adslab if site.surface_properties == 'adsorbate']:
            print('removed all adsorbates! original coverage, ', coverage)
            if invalid:
                return 'invalid'
            else:
                return False
        else:
            print('removed %s adsorbates! original coverage' %(len(indices_to_remove)/len(molecule)), coverage)
            return adslab
        
    def generate_random_adsorption_structure(self, molecule, n_adslabs, max_coverage=None,
                                             axis=[0, 0, 1], n_rotations=4, angles_list=[]):
        """
        Sample n_adslabs for all possible adsorbate configs on an MXide surface. Basic algorithm:
            1. Pick a random coverage
            2. Pick a random set of adsorbate positions (depending on the coverage sampled)
            3. Pick a random angle to rotation about the axis of the adsorbate (360/n_rotations
                possible angles)
            4. Combine to create a unique adsorbate configuration on the surface
            5. Repeat n_adslabs amount of times. There are so many possible configurations with
                this many variables that it is highly unlikely we will sample the exact same configuration

        Args:
            Molecule (pmg Molecule or list of two Molecules): Two options:
                1. We provide a pymatgen Molecule object representing the adsorbate we are interested in.
                2. If we want to get MVK-like adsorbate positions, we need a list with a pair of molecules.
                    The first Molecule in the list is the actual adsorbate on the surface. The second
                    molecule designates how we want this adsorbate to bind with the X lattice position on
                    the surface. e.g. We want to adsorb O on the surface of M-O to form O2 with the terminated
                    O-lattice positions. Then O will be the first item and O2 will be the second item in the list
            n_adslabs (int): Arguments for AdsorbateSiteFinder class
            max_coverage (list or 'all' or 'saturate'): List of number of adsorbates
                we want on surface (basically list of possible coverages to consider).
                If 'all', get all possible coverages. If 'saturate', get fully
                coordinated metals on surface.
            seed (str): Bulk-element to 'adsorb', typically the non-metal component of MXide
            axis (float): minimum length and width of the slab, only used
                if repeat is None
            n_rotations (3-tuple or list): repeat argument for supercell generation
        """
        start = time.time()
        n = len(molecule[0]) if type(molecule).__name__ == 'list' else len(molecule)
        n_rotations = 1 if n == 1 else n_rotations
        angles_list = [(2 * np.pi / n_rotations) * i for i in range(n_rotations)] if not angles_list else angles_list
        
        if type(molecule).__name__ != 'list':
            if hasattr(molecule, 'dimer_coupling'):
                try:
                    rot_and_pos_combs = self.get_all_rot_and_pos_combs(molecule, axis=axis,
                                                                       angles_list=angles_list, as_list=True)
                except IndexError:
                    return []
            else:
                rot_and_pos_combs = self.get_all_rot_and_pos_combs(molecule, axis=axis,
                                                                   angles_list=angles_list, as_list=True)
        else:
            rot_and_pos_combs = self.get_all_rot_and_pos_combs(molecule, axis=axis,
                                                               angles_list=angles_list, as_list=True)
            
        end = time.time()
        m = molecule[0] if type(molecule).__name__ == 'list' else molecule
        print('found %s rot/pos combos in %.3f secs for %s' 
              %(len(rot_and_pos_combs), end-start, m.composition))
        
        max_cov = len(self.MX_adsites) + len(self.mvk_adsites) \
            if type(molecule).__name__ == 'list' else len(self.MX_adsites)
        max_coverage = max_cov if max_coverage == None else max_coverage
        coverage_list = range(max_coverage)
        if len(rot_and_pos_combs) == 0:
            if self.verbose:
                print('no viable adsorption sites')
            return []
        
        # Im gonna assume there are so many possible configs given the parameters,
        # its highly unlikely we run into the same symmetrically equivalent config
        random_adslabs = []
        while len(random_adslabs) != n_adslabs:
            start = time.time()
            
            found_comb_rot_and_pos = False
            while not found_comb_rot_and_pos:
                try:
                    if len(self.MX_adsites) == 0 and len(self.mvk_adsites) == 0:
                        print('NO ADSORPTION SITES FOUND!')
                        return []
                    coverage = self.random.sample(coverage_list, 1)[0] + 1
                except ValueError:
                    raise Exception('ERROR, coverage_list=%s' %(coverage_list))

                if len(rot_and_pos_combs) < coverage:
                    coverage = len(rot_and_pos_combs)

                if hasattr(molecule, 'dimer_coupling'):
                    print('dimer_coupling detected')
                    coverage = 1

                comb_rot_and_pos = self.random.sample(rot_and_pos_combs, coverage)
                if hasattr(molecule, 'dimer_coupling'):
                    break
                if coverage not in [1, len(self.MX_adsites)+len(self.mvk_adsites)]:
                    # skip any combination where two pos are within min adsorbate distance of each other
                    if any([self.is_pos_too_close(rot_pos_pair[0][1], rot_pos_pair[1][1]) \
                            for rot_pos_pair in itertools.combinations(comb_rot_and_pos, 2)]):
                        continue
                found_comb_rot_and_pos = True
            
            # get all the adsorbate sites so we can get symmetrically distinct random combos
            adsites, ads_coords = [], []
            for translated_molecule, ads_coord, p in comb_rot_and_pos:
                molecule = translated_molecule.copy()
                molecule.add_site_property('ads_coord', [ads_coord]*len(molecule))
                for site in molecule:
                    adsites.append(PeriodicSite(site.species, ads_coord + site.coords, 
                                                self.slab.lattice, properties=site.properties, 
                                                coords_are_cartesian=True))
                    ads_coords.append(ads_coord)
            
            adslab = self.adslab_from_adsites(adsites, coverage)
            if adslab == 'invalid':
                break
            if not adslab:
                print('skipping')
                continue    
            else:                
                adslab = label_sites(adslab)
                setattr(adslab, 'ads_coords', ads_coords)
                random_adslabs.append(adslab)
                
                end = time.time()
                print('found adslab in %.3f secs' %(end-start))

        return random_adslabs
    
    def get_distinct_partition_combinations(self, coverage):

        # first identify how many surface sites are there per partition
        distinct_partitions = []
        for p in self.slab.site_properties['supercell']:
            if p not in distinct_partitions:
                distinct_partitions.append(p)
        adsites_per_partition = (len(self.MX_adsites)+len(self.mvk_adsites))/len(distinct_partitions)
        
        # now determine the number of distinct combinations of 
        # partitions to accomadate the number of adsorbates in this coverage
        distinct_partition_combos = []
        for i in range(0, len(distinct_partitions)):
            if (i+1)*adsites_per_partition < coverage or i+1 > coverage:
                continue 
            
            # To find out if adsorbates on a combination of partitions is 
            # symmetrically distinct, we build a defective slabs with the 
            # surface sites of the selected partitions of the combination removed 
            partition_combos = []
            for c in itertools.combinations(distinct_partitions, i+1):
                defect_slab = self.slab.copy()
                to_remove = []
                for nsite, site in enumerate(defect_slab):
                    if 'Top' == site.site_type and site.supercell in c:
                        to_remove.append(nsite)
                defect_slab.remove_sites(to_remove)
                setattr(defect_slab, 'partition', c)
            partition_combos.append(defect_slab)
            distinct_partition_combos.extend([g[0].partition for g in 
                                              self.sm.group_structures(partition_combos)])
        return distinct_partition_combos        

    def generate_adsorption_structures(self, molecule, coverage_list, radius=2.4, bond={},
                                       axis=[0, 0, 1], n_rotations=4, angles_list=[], consistent_rotation=False):
        """
        Sample n_adslabs for all possible adsorbate configs on an MXide surface. Basic algorithm:
            1. Pick a random coverage
            2. Pick a random set of adsorbate positions (depending on the coverage sampled)
            3. Pick a random angle to rotation about the axis of the adsorbate (360/n_rotations
                possible angles)
            4. Combine to create a unique adsorbate configuration on the surface
            5. Repeat n_adslabs amount of times. There are so many possible configurations with
                this many variables that it is highly unlikely we will sample the exact same configuration

        Args:
            Molecule (pmg Molecule or list of two Molecules): Two options:
                1. We provide a pymatgen Molecule object representing the adsorbate we are interested in.
                2. If we want to get MVK-like adsorbate positions, we need a list with a pair of molecules.
                    The first Molecule in the list is the actual adsorbate on the surface. The second
                    molecule designates how we want this adsorbate to bind with the X lattice position on
                    the surface. e.g. We want to adsorb O on the surface of M-O to form O2 with the terminated
                    O-lattice positions. Then O will be the first item and O2 will be the second item in the list
            n_adslabs (int): Arguments for AdsorbateSiteFinder class
            max_coverage (list or 'all' or 'saturate'): List of number of adsorbates
                we want on surface (basically list of possible coverages to consider).
                If 'all', get all possible coverages. If 'saturate', get fully
                coordinated metals on surface.
            seed (str): Bulk-element to 'adsorb', typically the non-metal component of MXide
            axis (float): minimum length and width of the slab, only used
                if repeat is None
            n_rotations (3-tuple or list): repeat argument for supercell generation
        """

        # for the case of dimer coupling, return nothing 
        # if not metal sites are available on the surface
        if len(self.surf_metal_sites) < 2 and type(molecule).__name__ != 'list' \
        and  'dimer_coupling' in molecule.site_properties.keys():
            return []

        # set n_rotation to 1 if monatomic to save time
        n = len(molecule[0]) if type(molecule).__name__ == 'list' else len(molecule)
        n_rotations = 1 if n == 1 else n_rotations
        angles_list = [(2 * np.pi / n_rotations) * i for i in range(n_rotations)] if not angles_list else angles_list
        
        # realign the initial position of the adsorbate such that 
        # it maximizes a user selected bond if 'bond' is designated
        if bond:
            molecule = self.maximize_adsorbate_bonds(molecule, bond, axis=axis)
        

        # retrieve a dict with xyz position as key pointing to a 
        # nested list of [rotated_molecule, adsite(xyz) and partition]
        rot_and_pos_combs = self.get_all_rot_and_pos_combs(molecule, axis=axis,
                                                           angles_list=angles_list)

        # determine list of different coverages for our list of adslabs.
        max_cov = len(self.MX_adsites) + len(self.mvk_adsites) \
                        if type(molecule).__name__ == 'list' else len(self.MX_adsites)
        if coverage_list == 'all':
            cov_list = np.array(range(1, max_cov + 1))
        elif coverage_list == 'saturated':
            cov_list = [max_cov]
        else:
            cov_list = copy.copy(coverage_list)
        if sum([len(rotpos_combs) for rotpos_combs in rot_and_pos_combs]) == 0:
            if self.verbose:
                print('no viable adsorption sites')
            return []

        all_adslabs = []
        for coverage in cov_list:
            adslabs = []
            if sum([len(rotpos_combs) for rotpos_combs in rot_and_pos_combs]) < coverage:
                coverage = sum([len(rotpos_combs) for rotpos_combs in rot_and_pos_combs])

            # only do combinations of rotation, position and coverage for a certain 
            # combination of partitions as all other iterations are degenerate
            partition_combos = self.get_distinct_partition_combinations(coverage)
            if self.verbose:
                print('coverage', coverage)

            for part_comb in partition_combos:

                # make a dictionary of adsite (xyz) keys pointing 
                # to n pairs of rotated molecule and xyz
                rot_and_pos_sets = {}
                for p in part_comb:
                    for adsite in rot_and_pos_combs.keys():
                        if rot_and_pos_combs[adsite][0][-1] == p:
                            rot_and_pos_sets[adsite] = rot_and_pos_combs[adsite]

                new_rot_and_pos_combs = get_reduced_combinations(rot_and_pos_sets, coverage, part_comb,
                                                                 consistent_rotation=consistent_rotation)
                if self.verbose:
                    print('partition_combos', part_comb, 'Total combos', len(new_rot_and_pos_combs))

                sorted_rot_and_pos_combs = sort_rotposcombs_by_angles(new_rot_and_pos_combs, angles_list)
                if self.verbose:
                    print('groups %s combinations into %s groups' %(len(new_rot_and_pos_combs), len(sorted_rot_and_pos_combs)))
                for nang, angs_count in enumerate(sorted_rot_and_pos_combs.keys()):

                    rot_and_pos_combs_subset = sorted_rot_and_pos_combs[angs_count]
                    # produce all adsites first so we can sift through redundancies
                    adsites_sets = []
                    for comb_rot_and_pos in rot_and_pos_combs_subset:

                        if coverage not in [1, len(self.MX_adsites)+len(self.mvk_adsites)]:
                            # skip any combination where two pos are within min adsorbate distance of each other
                            if any([self.is_pos_too_close(rot_pos_pair[0][1], rot_pos_pair[1][1]) \
                                    for rot_pos_pair in itertools.combinations(comb_rot_and_pos, 2)]):
                                continue

                        adsites = []
                        for translated_molecule, ads_coord, p in comb_rot_and_pos:
                            molecule = translated_molecule.copy()
                            molecule.add_site_property('ads_coord', [ads_coord]*len(molecule))
                            for site in molecule:
                                adsites.append(PeriodicSite(site.species, ads_coord + site.coords, 
                                                            self.slab.lattice, properties=site.properties, 
                                                            coords_are_cartesian=True))
                        adsites_sets.append(adsites)
                    
                    sorted_adsites_by_bonds = self.sort_adsites_by_bonds(adsites_sets, radius)
                    if self.verbose:
                        print('grouped %s sites to %s' %(len(adsites_sets), len(sorted_adsites_by_bonds.keys())))

                    count_tot, count_reduced = 0, 0
                    reduced_adsites_sets = []
                    for bonds in sorted_adsites_by_bonds.keys():
                        # get rid of all symmetrically equivalent adsorbate configs 
                        reduced_adsites_sets.extend(self.symm_reduce(sorted_adsites_by_bonds[bonds]))
                    if self.verbose:
                        print('symmetrically reduced to %s adslabs from %s' %(len(reduced_adsites_sets), len(adsites_sets)))

                    for adsite_set in reduced_adsites_sets:
                        adslab = self.adslab_from_adsites(adsite_set, coverage)
                        if not adslab:
                            continue
                        elif adslab == 'invalid':
                            continue
                        else:
                            adslab = label_sites(adslab)
                            adslabs.append(adslab)
                    if self.verbose:
                        print('sorting %s adslabs, completed %s/%s iterations' %(len(adslabs), nang, 
                                                                                 len(sorted_rot_and_pos_combs.keys())))
                all_adslabs.extend(adslabs)

        return all_adslabs
    
    def symm_reduce(self, adsites_set, threshold=1e-6):
        """
        Reduces the set of adsorbate sites by finding removing
        symmetrically equivalent duplicates
        Args:
            adsites_set: List of set of adsorbate sites on the slab. Each set 
            represents the cartesian coordinates of all adsorbate atoms on the surface
            threshold: tolerance for distance equivalence, used
                as input to in_coord_list_pbc for dupl. checking
        """
        surf_sg = SpacegroupAnalyzer(self.slab, 0.1)
        symm_ops = surf_sg.get_symmetry_operations()    
        # skip any symmops that operate outside the xy plane
        surf_symm_ops = []
        for op in symm_ops:
            if all((op.rotation_matrix[2] == [0, 0, 1])) and op.translation_vector[2] == 0:
                surf_symm_ops.append(op)
        
        # Convert to fractional
        coords_set = [[self.slab.lattice.get_fractional_coords(adsite.coords) 
                       for adsite in adsites] for adsites in adsites_set]

        unique_coords, unique_coords_species, unique_adsites = [], [], []
        for i, coords in enumerate(coords_set):
            # coords is a set of coordinates corresponding to all adsorbates on a slab
            incoord = False
            for op in surf_symm_ops:
                for ui, done_coords in enumerate(unique_coords):
                    one_to_one_coord_map = []
                    
                    op_coords = op.operate_multi(coords)
                    for ii, coord in enumerate(op_coords):
                        adsorbate = adsites_set[i][ii].species_string
                        try:
                            # see if theres a position in one of the 
                            # unique sets that matches the current one
                            inds = coord_list_mapping_pbc(coord, done_coords)
                            if adsorbate != unique_coords_species[ui][inds[0]]:
                                # check if the adsorbate species of this coordinate matches 
                                # the species we are comparing to. If it doesn't match, then skip
                                break
                            if inds[0] not in one_to_one_coord_map:
                                one_to_one_coord_map.append(inds[0])

                        except ValueError:
                            # if not one of the coordinates is in the set of unique sets, then we are 
                            # not getting a 1-to-1 mapping. Move on to the next symmetry operation
                            break

                    if len(one_to_one_coord_map) == len(coords):
                        # check if theres a one-to-one matching of coordinates
                        incoord = True
                        break

                if incoord:
                    break
            if not incoord:
                unique_coords.append(coords)           
                unique_coords_species.append([adsite.species_string for adsite in adsites_set[i]])
                unique_adsites.append(adsites_set[i])

        return unique_adsites
    
    def sort_adsites_by_bonds(self, adsites_set, radius):
        """
        Simple code for sorting a list of adsorption sites in a slab based 
        on their local environment defined by element and bondlength within a given radius.

        adslabs ([pmg Slab]): list of adsorbed slabs
        bonds (dict): Dict indicating the type of bond to sort by. Dict 
            has three items, the atomic species of the adsorbate, the atomic species the 
            adsorbate will form a bond with, and the bondlength +- blenght_tol e.g. 
            {'adsorbate': 'H', 'other': 'O', 'blength': 1.4} will sort all adslabs based 
            on the number of H-O bonds 1.4Å (+- the tol). Sorted from least amount of bonds to most
        blength_tol (float): tolerance +- the blength

        return dict of adslabs but sorted by number of bonds as keys
        """

        clean = self.slab.copy()
        sorted_adsites_set = {}
        for i, adsites in enumerate(adsites_set):
        
            bond_count = []
            for ads in adsites:
                neighbors = clean.get_neighbors(ads, radius)    
                for nn in neighbors:
                    if nn.distance(ads) < radius:
                        bond_count.append('%s-%s%.3f' %(ads.species_string,
                                                        nn.species_string, nn.distance(ads)))

            bond_count = tuple(sorted(bond_count))
            if bond_count not in sorted_adsites_set.keys():
                sorted_adsites_set[bond_count] = []
            sorted_adsites_set[bond_count].append(adsites_set[i])

        return sorted_adsites_set 
    
    def maximize_adsorbate_bonds(self, molecule, bonds, axis=[0,0,1]):

        adslab = self.generate_adsorption_structures(molecule, 'saturated', 2.4, n_rotations=1)[0]
        angles, ads_dists = [], []
        for nsite, site in enumerate(adslab):
            
            if site.species_string == bonds['adsorbate'] and site.surface_properties == 'adsorbate':
                dummy_adsite = PeriodicSite('N', site.properties['ads_coord'], adslab.lattice, coords_are_cartesian=True)
                nn = [neighbor for neighbor in adslab.get_neighbors(dummy_adsite, 5) \
                      if neighbor.species_string == bonds['other'] and round(neighbor.nn_distance, ndigits=4) != 0]
                dists = [n.nn_distance for n in nn]
                closest_sites = [nn[i] for i, d in enumerate(dists) if round(d, ndigits=4) == round(min(dists), ndigits=4)]

                # next calculate the angle of rotation about the give axis of rotation needed to minimize the distance between
                # the adsorbate in question by solving the angle between the adsorbate, dummy_adsite and neighbor site in 2D
                for ncsites, csites in enumerate(closest_sites):
                    rotated_ads_on_slab = adslab.copy()
                    coords_in_2d = [np.array([site.coords[0], site.coords[1], 0]),
                                    np.array([dummy_adsite.coords[0], dummy_adsite.coords[1], 0]), 
                                    np.array([csites.coords[0], csites.coords[1], 0])]

                    ba, bc = coords_in_2d[0]-coords_in_2d[1], coords_in_2d[2]-coords_in_2d[1]
                    cosa = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(round(cosa, ndigits=3))
                    rotated_ads_on_slab.rotate_sites([nsite], angle, adslab.lattice.matrix[2], dummy_adsite.coords)
                    nn = rotated_ads_on_slab.get_neighbors(site, 5)
                    ads_dists.append(min([n.nn_distance for n in nn]))
                    angles.append(angle)
                    
        molecule_rot = molecule.copy()
        molecule_rot.rotate_sites(theta=angles[ads_dists.index(min(ads_dists))], axis=axis)
        
        return molecule_rot
        
def get_reduced_combinations(rot_and_pos_sets, coverage, partition_combo, consistent_rotation=False):
    '''
    Helper function to get a set of rotated molecules and positions 
    of a certain coverage with the input set being limited to the  
    distinct combinations of partitions in a super slab cell. Uses  
    product enumeration to avoid iterating through combinations  
    containing duplicate adsites
    
    rot_and_pos_sets: dictionary with key being xyz, value is 
    list of rotated molecules. This dictionary is generated 
    from which ever partition combination 
    '''
    
    all_rot_and_pos_combs = []
    # enumerate all combinations of xyz
    
    for c in itertools.combinations(rot_and_pos_sets.keys(), coverage):
        # skip any combinations of xyz where all partitions we are considering are not present
        if not all([p in [rot_and_pos_sets[xyz][0][2] for xyz in c] for p in partition_combo]):
            continue
            
        if consistent_rotation:
            # faster algo that only considers combos where adsorbates rotated consistently
            for i, d in enumerate(rot_and_pos_sets[c[0]]):
                rot_and_pos_combs = []
                for xyz in c:
                    rot_and_pos_combs.append(rot_and_pos_sets[xyz][i])
                all_rot_and_pos_combs.append(rot_and_pos_combs)
            
        else:
            # enumerate all combinations of n rotations where n is number of xyz 
            # positions (coverage), each of which can be rotated several times. 
            # The first entry in the product function should be range from 0 to 
            # n where n is the number of rotations. Second number for repeat is the coverage.
            for sites_indices in itertools.product(range(0, len(rot_and_pos_sets[c[0]])),
                                                   repeat=coverage):
                rot_and_pos_combs = []
                for i, xyz in enumerate(c):
                    rot_and_pos_combs.append(rot_and_pos_sets[xyz][sites_indices[i]])
                all_rot_and_pos_combs.append(rot_and_pos_combs)
            
    return all_rot_and_pos_combs


from pymatgen.core.structure import Lattice

def make_superslab_with_partition(slab, scaling_matrix):
    """
    Makes a supercell. Allowing to have sites outside the unit cell. Also adds site property, 'supercell', 
    distinguishing each additional unit of the original cell that was added with a unique index
    Args:
        scaling_matrix: A scaling matrix for transforming the lattice
            vectors. Has to be all integers. Several options are possible:
            a. A full 3x3 scaling matrix defining the linear combination
               the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
               1]] generates a new structure with lattice vectors a' =
               2a + b, b' = 3b, c' = c where a, b, and c are the lattice
               vectors of the original structure.
            b. An sequence of three scaling factors. E.g., [2, 1, 1]
               specifies that the supercell should have dimensions 2a x b x
               c.
            c. A number, which simply scales all lattice vectors by the
               same factor.
    Returns:
        Supercell structure. Note that a Structure is always returned,
        even if the input structure is a subclass of Structure. This is
        to avoid different arguments signatures from causing problems. If
        you prefer a subclass to return its own type, you need to override
        this method in the subclass.
    """
    scale_matrix = np.array(scaling_matrix, np.int16)
    if scale_matrix.shape != (3, 3):
        scale_matrix = np.array(scale_matrix * np.eye(3), np.int16)
    new_lattice = Lattice(np.dot(scale_matrix, slab._lattice.matrix))

    f_lat = lattice_points_in_supercell(scale_matrix)
    c_lat = new_lattice.get_cartesian_coords(f_lat)

    new_sites = []

    for site in slab:
        for i, v in enumerate(c_lat):
            new_prop = site.properties.copy()
            new_prop['supercell'] = i
            s = PeriodicSite(
                site.species,
                site.coords + v,
                new_lattice,
                properties=new_prop,
                coords_are_cartesian=True,
                to_unit_cell=False,
                skip_checks=True,
            )
            new_sites.append(s)

    new_slab = Structure.from_sites(new_sites)

    return Slab(new_slab.lattice, new_slab.species, new_slab.frac_coords, 
                slab.miller_index, slab.oriented_unit_cell, slab.shift, 
                slab.scale_factor, site_properties=new_slab.site_properties)

def sort_by_bonds(adslabs, bond):
    """
    Simple code for sorting a list of adsorbed slabs from MXideAdsorbateGenerator 
    based on the number of designated bond types in the bonds list
    
    adslabs ([pmg Slab]): list of adsorbed slabs
    bonds (dict): Dict indicating the type of bond to sort by. Dict 
        has three items, the atomic species of the adsorbate, the atomic species the 
        adsorbate will form a bond with, and the bondlength +- blenght_tol e.g. 
        {'adsorbate': 'H', 'other': 'O', 'blength': 1.4} will sort all adslabs based 
        on the number of H-O bonds 1.4Å (+- the tol). Sorted from least amount of bonds to most
    blength_tol (float): tolerance +- the blength
    
    return dict of adslabs but sorted by number of bonds as keys
    """
    
    sorted_adslabs = {}
    for adslab in adslabs:
        # find all designated adsorbates
        adsites = [site for site in 
                   adslab if site.surface_properties == 'adsorbate' \
                   and site.species_string == bond['adsorbate']]
        bond_count = []
        for ads in adsites:
            neighbors = adslab.get_neighbors(ads, bond['blength'])
            for nn in neighbors:
                if nn.species_string == bond['other']:
                    if nn.distance(ads) < bond['blength']:
                        bond_count.append(nn.distance(ads))
           
        bond_count = tuple(sorted(bond_count))
        if bond_count not in sorted_adslabs.keys():
            sorted_adslabs[bond_count] = []
        sorted_adslabs[bond_count].append(adslab)
        
    return sorted_adslabs    

def sort_rotposcombs_by_angles(rot_and_pos_combs, angles_list):
    """
    Sort all combinations of rotation and adsorbate postion based on the
        rotation angles. Makes it faster to identify redundant adsites
    """
    
    rotation_set = [round(a, ndigits=2) for a in angles_list]
    sorted_rot_and_pos_combs_by_angles = {}
    for rot_and_pos_comb in rot_and_pos_combs:
        rotation_count = {ang: 0 for ang in rotation_set}
        for rot_and_pos in rot_and_pos_comb:
            rotation_count[rot_and_pos[0].deg] += 1
        rotation_count = tuple(rotation_count.values())
        if rotation_count not in sorted_rot_and_pos_combs_by_angles.keys():
            sorted_rot_and_pos_combs_by_angles[rotation_count] = []
        sorted_rot_and_pos_combs_by_angles[rotation_count].append(rot_and_pos_comb)
        
    return sorted_rot_and_pos_combs_by_angles

def label_sites(slab):
    
    """
    Distinguish surface sites and bulk-like sites. Must take in a slab with 
        surface_properties (see AdsorptionSiteFinder or MXideAdsorbateGenerator).
        Identifies surface sites as being on the top or bottom surface
    """
    slab = slab.copy()
    is_surfsite_list = []
    for i, site in enumerate(slab):
        if site.frac_coords[2] > slab.center_of_mass[2] and \
        site.surface_properties in ['surface', 'adsorbate']:
            is_surfsite_list.append('Top')
        elif site.frac_coords[2] < slab.center_of_mass[2] and \
        site.surface_properties in ['surface', 'adsorbate']:
            is_surfsite_list.append('Bottom')
        else:
            is_surfsite_list.append('Bulk')
    slab.add_site_property('site_type', is_surfsite_list)

    return slab

def get_repeat_from_min_lw(slab, min_lw):
    """
    Modified version of algorithm from adsorption.py for determining the super cell 
        matrix of the slab given min_lw. This will location the smallest super slab 
        cell with min_lw by including square root 3x3 transformation matrices
    """
    
    xlength = np.linalg.norm(slab.lattice.matrix[0])
    ylength = np.linalg.norm(slab.lattice.matrix[1])
    xrep = np.ceil(min_lw / xlength)
    yrep = np.ceil(min_lw / ylength)
    rtslab = slab.copy()
    rtslab.make_supercell([[1,1,0], [1,-1,0], [0,0,1]])
    rt_matrix = rtslab.lattice.matrix
    xlength_rt = np.linalg.norm(rt_matrix[0])
    ylength_rt = np.linalg.norm(rt_matrix[1])
    xrep_rt = np.ceil(min_lw / xlength_rt)
    yrep_rt = np.ceil(min_lw / ylength_rt)

    xrep = xrep*np.array([1,0,0]) if xrep*xlength < xrep_rt*xlength_rt else xrep_rt*np.array([1,1,0]) 
    yrep = yrep*np.array([0,1,0]) if yrep*ylength < yrep_rt*ylength_rt else yrep_rt*np.array([1,-1,0]) 
    zrep = [0,0,1]
    return [xrep, yrep, zrep]
