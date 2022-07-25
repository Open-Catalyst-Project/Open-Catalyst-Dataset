import numpy as np
from pymatgen.core.structure import Molecule


"""
Quick manuel enumeration of a few basic molecules for
OER. Rotational degree of freedom handled by MXide_adsorption.
"""

# Molecules

# O2
O2 = Molecule(["O", "O"], [np.array([0, 0.99232, 0.61263])/np.linalg.norm(np.array([0, 0.99232, 0.61263]))*1.208, 
                           [0,0,0]])

# O (can bind to surface oxygen to form O2)
Ox = Molecule(["O"], [[0,0,0]])
O2_like = O2.copy()
O2_like.add_site_property('anchor', [False, True])

# Oxo-coupling
Oxo_coupling = Molecule(["O", "O"], [(1.6815, 0, 0,), (0,0,0)])
Oxo_coupling.add_site_property('dimer_coupling', [True, True])
setattr(Oxo_coupling, 'dimer_coupling', True)


# OOH
OOH_up = Molecule(["O","O","H"], [[0, 0, 0], [-1.067, -0.403, 0.796],[-0.696, -0.272, 1.706]])
OOH_down = Molecule(["O","O","H"], [[0,0,0], [-1.067, -0.403, 0.796], [-1.84688848, -0.68892498, 0.25477651]])

# OH (can bind to surface oxygen to form OOH)
OH = Molecule(["O","H"], [[0,0,0], 
                          np.array([0, 0.99232, 0.61263])/np.linalg.norm(np.array([0, 0.99232, 0.61263]))*1.08540])
OOH_up_OH_like = OOH_up.copy()
OOH_up_OH_like.add_site_property('anchor', [True, False, False])
OH_pair = [OH, OOH_up_OH_like]

# H2O
H2O = Molecule(['H', 'H', 'O'], [[2.226191, -9.879001, 2.838300],
                                 [2.226191, -8.287900, 2.667037],
                                 [2.226191, -9.143303, 2.156037]])

# CO (can bind to surface oxygen to form CO2)
CO = Molecule(["C", "O"], [[0,0,1.43], [0,0,0]])
CO.rotate_sites(theta=45, axis=[1,0,0])
CO2 = Molecule(["C", "O", "O"], [[0,0,0], [-0.6785328, -0.6785328, -0.6785328],
                       [0.6785328, 0.6785328, 0.6785328]])
CO2_like = CO2.copy()
CO2_like.add_site_property('anchor', [False, True, False])

# H (can bind to surface oxygen to form OH)
Hx = Molecule(["H"], [[0,0,0]])
OH_like = OH.copy()
OH_like.add_site_property('anchor', [True, False])

# N (can bind to surface oxygen to form NO)
Nx = Molecule(["N"], [[0,0,0]])
NO_like = Molecule("NO", [[0,0,1.16620], [0,0,0]])
NO_like.rotate_sites(theta=45, axis=[1,0,0])
NO_like.add_site_property('anchor', [False, True])

# C (can bind to surface oxygen to form CO)
Cx = Molecule(["C"], [[0,0,0]])
CO_like = CO.copy()
CO_like.add_site_property('anchor', [False, True])

# All adsorbates
adslist = [[Ox, O2_like], [Hx, OH_like], [Nx, NO_like], [Cx, CO_like],
           # monatomic adsorption of O, H, N and C can form O2,
           # OH, NO and CO with lattice positions respectively
           OH_pair, O2, [CO, CO2_like], Oxo_coupling, 
           H2O]
OOH_list = [OOH_up, OOH_down]