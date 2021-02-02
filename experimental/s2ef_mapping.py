import ase.io


old_systems = glob.glob("path_to_old_systems/*.traj")
new_system_path = "path_to_new_systems"

system_mappings = {}
for system in old_system:
    randomid = os.path.basename(system)[:-5]
    old_images = ase.io.read(system, ":")
    old_energies = [image.get_potential_energy() for image in old_iamges]

    new_images = ase.io.read(f"{new_system_path}/{randomid}.traj", ":")
    new_energies = [image.get_potential_energy() for image in new_iamges]

    dict_mapping = {}
    for old_idx, energy in enumerate(old_energies):
        new_idx = np.where(new_energies == energy)[0]
        assert len(new_idx) == 1 # ensure no duplicates
        dict_mapping[old_idx] = new_idx.item()

    system_mappings[randomid] = dict_mapping
