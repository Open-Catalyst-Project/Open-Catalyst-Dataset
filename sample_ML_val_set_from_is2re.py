## Import Packages
import pandas as pd
import pickle
import random
from glob import glob

# Get the is2re Data & Exclude Some MPID that not appropriate bulks (S, P, H, etc)
mpids_to_exclude = ['mp-672234', 'mp-632250', 'mp-754514', 'mp-570747', 'mp-12103', 'mp-25',
                    'mp-672233', 'mp-568584', 'mp-154', 'mp-999498', 'mp-14', 'mp-96', 'mp-1080711',
                    'mp-1008394', 'mp-22848', 'mp-160', 'mp-1198724']

# put ocdata is2re calculations in a df
ocd_df = pd.read_pickle('_cache/ocd_data_with_formula.pkl')
ocd_df = ocd_df[~ ocd_df['mpid'].isin(mpids_to_exclude)]
ocd_df['n_elems'] = ocd_df.apply(lambda x: len(x['stoichiometry'].keys()), axis=1)

n_cat_elems_weights = {1: 0.1, 2: 0.6, 3: 0.3}
ocd_df['n_elems_weights'] = ocd_df.apply(lambda x: n_cat_elems_weights[x['n_elems']], axis=1)

# Get subset of the /materials/surface/adsorbate combo
# based on Val random ID and bulk compositions
def sample_val_set_by_weight(file_name, df, n, weights):
    """
    Args:
        file (str)    file name of the file that has the list of
                      random ID's of the ocp validation set.
        df            A pandas Dataframe that has all the ocp is2re data and their
                      material/surface/adsorbate combo (compiled based on the pickle file Sid provided).
        n (int)       Number of samples selected.
        weights (str) The name of df column the sampling weights are based on.

    Returns:
        sampled_df    The n ocp is2re data and their material/surface/adsorbate sampled (in a dataframe).
    """
    with open(file_name, 'r') as f:
        random_ids = f.read().split("\n")[:-1]
    val_df = df.loc[df.ID.isin(random_ids)]
    sampled_df = val_df.sample(n=n, weights=weights)
    return sampled_df

sampled_df = pd.DataFrame()
for file in glob('_cache/ocp_filtered_val_sets/*'):
    sampled_df = sampled_df.append(sample_val_set_by_weight(file, ocd_df, 250, 'n_elems_weights'))


# Make Input Files and Save
bulk_ads = []
bulk_slab_ads = []
for idx, row in sampled_df.iterrows():
    bulk_ads.append((row['mpid'], row['adsorbate']))
    bulk_slab_ads.append((row['mpid'], row['miller'], row['shift'], row['top'], row['adsorbate']))
 
# pickle.dump(bulk_ads, open('_cache/filtered_val_sets/bulk_adsorbate_1000.pkl', 'wb'))
# pickle.dump(bulk_ads, open('_cache/filtered_val_sets/bulk_surface_adsorbate_1000.pkl', 'wb'))
