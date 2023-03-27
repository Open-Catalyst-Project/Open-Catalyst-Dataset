import ase
import numpy as np
import scipy

from ocdata.core import Adsorbate, Surface


class Structure:
    """
    Initializes an adsorbate-catalyst system for a given Adsorbate and Surface.
    """

    def __init__(
        self,
        surface: Surface,
        adsorbate: Adsorbate,
        num_sites: int = 100,
        num_augmentations_per_site: int = 1,
        added_z: int = 2,
    ):
        self.surface = surface
        self.adsorbate = adsorbate
        self.num_sites = num_sites
        self.num_augmentations_per_site = num_augmentations_per_site

        self.sites = self.get_binding_sites(num_sites, added_z)
        self.adslabs = self.place_adsorbate_on_sites(
            self.sites, num_augmentations_per_site
        )
        self.adslabs = self.filter_unreasonable_adslabs(self.adslabs)

    def get_binding_sites(self, num_sites: int, added_z: int = 2):
        """
        Returns `num_sites` sites given the surface atoms' positions.
        """
        surface_atoms_idx = [
            i for i, atom in enumerate(self.surface.atoms) if atom.tag == 1
        ]
        surface_atoms_pos = self.surface.atoms[surface_atoms_idx].positions

        dt = scipy.spatial.Delaunay(surface_atoms_pos[:, :2])
        simplices = dt.simplices
        num_sites_per_triangle = int(np.ceil(num_sites / len(simplices)))
        all_sites = []
        for tri in simplices:
            triangle_positions = surface_atoms_pos[tri]
            sites = get_random_sites_on_triangle(
                triangle_positions, num_sites_per_triangle, added_z
            )
            all_sites += sites
        np.random.shuffle(all_sites)
        return all_sites[:num_sites]

    def place_adsorbate_on_site(self, site: np.ndarray):
        """
        Place the adsorbate at the given binding site.
        """
        adsorbate_c = self.adsorbate.atoms.copy()
        surface_c = self.surface.atoms.copy()

        # Translate adsorbate to binding site.
        com = adsorbate_c.get_center_of_mass()
        translation_vector = site - com
        adsorbate_c.translate(translation_vector)

        # Rotate adsorbate along x, y, z.
        angles = np.random.uniform(0, 360, 3)
        adsorbate_c.rotate(angles[0], v="x", center="COM")
        adsorbate_c.rotate(angles[1], v="y", center="COM")
        adsorbate_c.rotate(angles[2], v="z", center="COM")

        # Combine adsorbate and surface, and set tags correctly.
        adslab = surface_c + adsorbate_c
        tags = [2] * len(adsorbate_c)
        final_tags = list(surface_c.get_tags()) + tags
        adslab.set_tags(final_tags)

        # Set pbc and cell.
        adslab.cell = surface_c.cell
        # Comment(@abhshkdz): Why don't we set PBC=True along all 3 axes? The
        # surface did have PBC along all 3 axes, right?
        adslab.pbc = [True, True, False]

        return (site, adslab)

    def place_adsorbate_on_sites(
        self, sites: list, num_augmentations_per_site: int = 1
    ):
        """
        Place the adsorbate at the given binding sites.
        """
        adslabs = []
        for site in sites:
            for _ in range(num_augmentations_per_site):
                adslabs.append(self.place_adsorbate_on_site(site))

        return adslabs

    def filter_unreasonable_adslabs(self, adslabs: list):
        """
        Filter out unreasonable adsorbate-surface systems.
        """
        filtered_adslabs = []
        for i in adslabs:
            site, adslab = i
            if is_adslab_reasonable(adslab):
                filtered_adslabs.append(i)
        return filtered_adslabs


def get_random_sites_on_triangle(
    vertices: np.ndarray,
    num_sites: int = 10,
    added_z: int = 0,
):
    """
    Sample `num_sites` random sites uniformly on a given 3D triangle.
    Following Sec. 4.2 from https://www.cs.princeton.edu/~funk/tog02.pdf.
    """
    assert len(vertices) == 3
    r1_sqrt = np.sqrt(np.random.uniform(0, 1, num_sites))[:, np.newaxis]
    r2 = np.random.uniform(0, 1, num_sites)[:, np.newaxis]
    sites = (
        (1 - r1_sqrt) * vertices[0]
        + r1_sqrt * (1 - r2) * vertices[1]
        + r1_sqrt * r2 * vertices[2]
    )
    # Comment(@abhshkdz): Okay to assume z is normal to the surface?
    sites[:, 2] += added_z
    return [i for i in sites]  # Hack to return a list of np.ndarray.


def is_adslab_reasonable(
    adslab: ase.Atoms,
):
    """
    Check if the adsorbate-surface system is reasonable.
    """
    # Comment(@abhshkdz): This is a placeholder for now.
    return True
