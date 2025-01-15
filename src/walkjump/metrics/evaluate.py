import pandas as pd
from walkjump.metrics import LargeMoleculeDescriptors
from tqdm import tqdm
import csv
import os
from walkjump.metrics import get_batch_descriptors

def get_descriptors_as_dict(sequence: str) -> dict:
    return {k: v for k, v in LargeMoleculeDescriptors.from_sequence(sequence).asdict().items()
            if k in set(LargeMoleculeDescriptors.descriptor_names())}

def rename_df(df: pd.DataFrame, prefix: str):
    df.rename({c: f"{prefix}_{c}" for c in df.columns}, inplace=True, axis=1)


# Load the generated sample.csv
df = pd.read_csv("data/poas.csv.gz")
print(df.head())
filename = "samplesED_32_dim_10"
sample_df_with_descriptors = pd.read_csv("data/" + filename + ".csv")
print(sample_df_with_descriptors.head())

tqdm.pandas(desc="heavy")
descriptor_df_heavy = pd.DataFrame.from_records(df.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
print(descriptor_df_heavy)

tqdm.pandas(desc="light")
descriptor_df_light = pd.DataFrame.from_records(df.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for light chains
print(descriptor_df_light)

rename_df(descriptor_df_heavy, "fv_heavy")
rename_df(descriptor_df_light, "fv_light")

ref_feats = pd.concat([descriptor_df_heavy, descriptor_df_light, df], axis=1)

print("Columns in sample_df_with_descriptors:", sample_df_with_descriptors.columns)
print("Columns in ref_feats:", ref_feats.columns)

tqdm.pandas(desc="sample heavy")
sample_df_with_descriptors_plus_heavy = pd.DataFrame.from_records(sample_df_with_descriptors.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
print(sample_df_with_descriptors_plus_heavy)

tqdm.pandas(desc="sample light")
sample_df_with_descriptors_plus_light = pd.DataFrame.from_records(sample_df_with_descriptors.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
print(sample_df_with_descriptors_plus_light)

rename_df(sample_df_with_descriptors_plus_heavy, "fv_heavy")
rename_df(sample_df_with_descriptors_plus_light, "fv_light")

sample_feats = pd.concat([sample_df_with_descriptors_plus_heavy, sample_df_with_descriptors_plus_light, sample_df_with_descriptors], axis=1)

print(sample_feats)

wasserstein_distances1, avg_wd1, total_wd1, prop_valid1 = get_batch_descriptors(sample_feats, ref_feats, "fv_heavy")
wasserstein_distances2, avg_wd2, total_wd2, prop_valid2 = get_batch_descriptors(sample_feats, ref_feats, "fv_light")

print(filename)
print(wasserstein_distances1, avg_wd1, total_wd1, prop_valid1)
print(wasserstein_distances2, avg_wd2, total_wd2, prop_valid2)

