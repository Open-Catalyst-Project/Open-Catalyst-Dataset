# Path to a database of bulks, organized as a dictionary with a unique integer
# as key and corresponding bulk tuple as value.
BULK_PKL_PATH = "/checkpoint/janlan/ocp/input_dbs/bulk_db_flat_2021sep20_ase3.21.pkl"

# Path to a folder of pickle files, each containing a list of precomputed
# slabs. The filename of each pickle is <bulk_index.pkl> where `bulk_index`
# is the index of the corresponding bulk in BULK_PKL_PATH.
PRECOMPUTED_SLABS_DIR_PATH = (
    "/checkpoint/janlan/ocp/input_dbs/precomputed_surfaces_2021Sep20/"
)

# Path to a database of adsorbates, organized as a dictionary with a unique
# integer as key and corresponding adsorbate tuple as value.
ADSORBATES_PKL_PATH = "ocdata/databases/pkls/adsorbates.pkl"
