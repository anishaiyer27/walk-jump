from walkjump.metrics import LargeMoleculeDescriptors
import pandas as pd
from tqdm import tqdm

def get_descriptors_as_dict(sequence: str) -> dict:
    return {k: v for k, v in LargeMoleculeDescriptors.from_sequence(sequence).asdict().items() if k in set(LargeMoleculeDescriptors.descriptor_names)}

def rename_df(df: pd.DataFrame, prefix: str):
    df.rename(columns={c: f"{prefix}_{c}" for c in df.columns}, inplace=True, axis=1)

df = pd.read_csv("./data/poas.csv.gz") # load your csv of paired sequences

print(df)

tqdm.pandas(desc="heavy")
descriptor_df_heavy = pd.DataFrame.from_records(df.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
descriptor_df_light = pd.DataFrame.from_records(df.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for light chains

rename_df(descriptor_df_heavy, "fv_heavy")
rename_df(descriptor_df_light, "fv_light")

ref_feats = pd.concat([descriptor_df_heavy, descriptor_df_light, df], axis=1)

from walkjump.sampling import walkjump

sample_df = pd.read_csv("./data/samples_2denoise.csv")

samp_descriptor_df_heavy = pd.DataFrame.from_records(sample_df.fv_heavy_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for heavy chains
samp_descriptor_df_light = pd.DataFrame.from_records(sample_df.fv_light_aho.str.replace("-", "").progress_apply(get_descriptors_as_dict).values) # make descriptors for light chains

rename_df(samp_descriptor_df_heavy, "fv_heavy")
rename_df(samp_descriptor_df_light, "fv_light")


sample_df_with_descriptors = pd.concat([sample_df, samp_descriptor_df_heavy, samp_descriptor_df_light], axis=1)


from walkjump.metrics import get_batch_descriptors

description_heavy = get_batch_descriptors(sample_df_with_descriptors, ref_feats, "fv_heavy")
description_light = get_batch_descriptors(sample_df_with_descriptors, ref_feats, "fv_light")

print(description_heavy)
print(description_light)