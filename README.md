[![CircleCI](https://dl.circleci.com/status-badge/img/gh/Open-Catalyst-Project/Open-Catalyst-Dataset/tree/refactor-v2.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/Open-Catalyst-Project/Open-Catalyst-Dataset/tree/refactor-v2)
[![codecov](https://codecov.io/gh/Open-Catalyst-Project/Open-Catalyst-Dataset/branch/master/graph/badge.svg?token=IZ7J729L6S)](https://codecov.io/gh/Open-Catalyst-Project/Open-Catalyst-Dataset)

# Open-Catalyst-Dataset

This repository hosts the input generation workflow used in the Open Catalyst Project.

## Setup

The easiest way to install prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following commands:

* Create a new environment, `vaspenv`, `conda create -n vaspenv python=3.9`
* Activate the newly created environment to install the required dependencies: `conda activate vaspenv`
* Install specific versions of Pymatgen and ASE: `pip install pymatgen==2023.1.20 ase==3.22.1`
* Clone this repo and install with: `pip install -e .`

## Workflow

The codebase supports the following workflow to generate adsorbate-slab configurations. A bulk atoms object may be provided, selected by `bulk_id` (i.e. `mp-30`), selected by its index in the database, or selected randomly.  Similarly, an adsorbate atoms object may be provided, selected by its SMILES string (i.e. `*H`), selected by its index in the database, or selected randomly. From the `Bulk` class, slabs may be enumerated which uses `pymatgen.core.surface.SlabGenerator`. This may be done three ways: (1) All slabs up to a specifiable miller index, (2) A random slab among those enumerated by (1) may be selected, or (3) a specific miller index may be enumerated. Once slab(s) have been enumerated, adsorbate placement may be performed. We use custom code inspired by what is implemented in pymatgen to do this. There are 3 placement modes: `random`, `heuristic`, and `random_site_heuristic_placement`. For all, a Delaunay meshgrid is constructed with surface atoms as nodes. For `heuristic` and `random_site_heuristic_placement` the adsorbate is randomly rotated in the z direction and provided some random "wobble", which is randomized tilt within a certain cone around the north pole.  For `heuristic` the adsorbate is placed on the node (atop), between 2 nodes (bridge) and in the center of the triangle (hollow). The adsorbate database includes information about which atoms are expected to bind. The binding atom is placed at the site. For `random_site_heuristic_placement` the binding atom is placed at the site as is for `heuristic`, but positions of the sites are randomly sampled along the Delaunay triangle. For `random`, the adsorbate is randomly rotated about its center of mass. The positions of the sites are randomly sampled along the Delaunay triangle and the center of mass of the adsorbate is placed at the site.

![Workflow image](ocdata_workflow.png)

## Use
Here is a simple example using the ocdata infra to place CO on Cu (1,1,1):
```
bulk_src_id = "mp-30"
adsorbate_smiles = "*CO"

bulk = Bulk(bulk_src_id_from_db = bulk_src_id, bulk_db_path = "your-path-here.pkl")
adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles, adsorbate_db_path = "your-path-here.pkl")
slabs = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,1,1))

# Perform heuristic placements
heuristic_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="heuristic")

# Perform random site, heuristic placements
random_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="random_site_heuristic_placement", num_sites = 100)
```
If you want to use a bulk and/or adsorbate that is not in the database here, you may supply your own atoms object like so:
```
bulk = Bulk(bulk_atoms = your_adsorbate_atoms)
adsorbate = Adsorbate(adsorbate_atoms=your_adsorbate_atoms)
slabs = Slab.from_bulk_get_all_slabs(bulk)

# Perform fully random placements
random_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="random", num_sites = 100)
```
If you would like to randomly choose a bulk, adsorbate, and slab:
```
bulk = Bulk()
adsorbate = Adsorbate()
slab = Slab.from_bulk_get_random_slab(bulk)

# Perform fully random placements
random_adslabs = AdsorbateSlabConfig(slab, adsorbate, mode="random", num_sites = 100)
```


## API supported

Generation may be facillitated using the `structure_generator.py` script. There are a number of options to configure input generation to suit different use cases. Below are the command line arguments and a few examples of use.

### Command Line Args

*Input Files:*
1. `--bulk_db` (required): path to the bulk database file
2. `--adsorbate_db`: path to the adsorbate database file - required if adsorbate placement is to be performed.
3. `--precomputed_slabs_dir`: path to the precomputed slab directory, which saves cost/time if the slabs for each bulk have already been enumerated.

*Material / adsorbate specification:*

Option 1: provide indices both must be provided to generate adsorbate-slab configurations, otherwise only slab enumeration will be performed.
1. `--adsorbate_index`: index of the desired adsorbate in the database file.
2. `--bulk_index`: index of the desired bulk
3. `--surface_index`: index of the desired surface

Option 2: provide a set of indices (one of the following)
1. `--indices_file`: a file containing strings with the following format `f"{adsorbate_idx}_{bulk_idx}_{surface_idx}"`. This will make slab enumeration and adsorbate slab enumeration be performed.
2. `--bulk_indices_file`: a file containing bulk indices, which will spark slab generation only

*Slab Enumeration:*
1. `--max_miller`: the max miller index of slabs to be generated (i.e. 1, 2, or 3)

*Adsorbate Placement:*
1. `--seed`: random seed for sampling/random sites generation.
2. `--heuristic_placements`: to be provided if heuristic placements are desired.
3. `--random_placements`: to be provided if random sites are desired. You may do both heurstic and random placements in the same run.
4. `--full_random_rotations`: to be provided in addition to `--random_placements` if fully random placements are desired, as opposed to small wobbles around x/y axis.
5. `--random_sites`: the number of sites per slab, which should be provided if `--random_placements` are used.
6. `--num_augmentations`: the number of random adsorbate configurations per site (defaults to 1).

*Multiprocessing, when given a file of indices:*
1. `--chunks`: for multi-node processing, number of chunks to split inputs across.
2. `--chunk_index`: for multi-node processing, index of chunk to process.
3. `--workers`: number of workers for multiprocessing within one job

*Outputs:*
1. `--output_dir`: directory to save outputs
2. `--no_vasp`: for VASP input files, only write POSCAR and do not write INCAR, KPOINTS, or POTCAR
3. `--verbose`: if detailed info should be logged

### Use
```
python structure_generator.py \
  --bulk_db databases/pkls/bulks.pkl \
  --adsorbate_db databases/pkls/adsorbates.pkl  \
  --output_dir outputs/ \
  --adsorbate_index 0 \
  --bulk_index 0 \
  --surface_index 0 \
  --heuristic_placements
```

```
python structure_generator.py \
  --bulk_db databases/pkls/bulks.pkl \
  --adsorbate_db databases/pkls/adsorbates.pkl  \
  --indices_file your_index_file.txt \
  --seed 0 \
  --random_placements \
  --random_sites 100

```


## Databases for bulk, adsorbate and precomputed surfaces

**Bulks**

A database of bulk materials taken from existing databases (i.e. Materials Project) and relaxed with consistent RPBE settings may be found in `ocdata/databases/pkls/bulks.pkl`. To preview what bulks are available, view the corresponding mapping between indices and bulks (bulk id and composition): https://dl.fbaipublicfiles.com/opencatalystproject/data/input_generation/mapping_bulks_2021sep20.txt

**Adsorbates**
A database of adsorbates may be found in `ocdata/databases/pkls/adsorbates.pkl`. Alternatively, it may be downloaded using the following link:
The latest version is https://dl.fbaipublicfiles.com/opencatalystproject/data/input_generation/adsorbate_db_2021apr28.pkl (MD5 checksum: `975e00a62c7b634b245102e42167b3fb`).
To preview what adsorbates are available, view the corresponding mapping between indices and adsorbates (SMILES): https://dl.fbaipublicfiles.com/opencatalystproject/data/input_generation/mapping_adsorbates_2020may12.txt


**Precomputed surfaces**

To speed up surface sampling from a chosen bulk material we precomputed surface enumerations up to miller index 2. These can be found here: https://dl.fbaipublicfiles.com/opencatalystproject/data/input_generation/precomputed_surfaces_2021sep20.tar.gz (5.5GB, MD5 checksum: `f56fe10f380d945d46a1bfab136a4834`)

Note that uncompressing this file will result in the folder `precomputed_surfaces_2021june11/`
(uncompressed size 18GB). It has 11k pickle files, with filename format `<zero-based-index>.pkl`, one for each of the bulk materials based on the bulk indices.


**Note**

OC20 was generated with an older version of the bulks and this repository. If you would like to reproduce that work exactly, see `README_legacy_OC20.md`

## Citation

The Open Catalyst 2020 (OC20) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode). Please cite the following paper in any research manuscript using the OC20 dataset:


```
@misc{ocp_dataset,
    title={The Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    author={Lowik Chanussot* and Abhishek Das* and Siddharth Goyal* and Thibaut Lavril* and Muhammed Shuaibi* and Morgane Riviere and Kevin Tran and Javier Heras-Domingo and Caleb Ho and Weihua Hu and Aini Palizhati and Anuroop Sriram and Brandon Wood and Junwoong Yoon and Devi Parikh and C. Lawrence Zitnick and Zachary Ulissi},
    year={2020},
    eprint={2010.09990},
    archivePrefix={arXiv}
}
```
