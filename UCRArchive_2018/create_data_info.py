import pandas as pd
import numpy as np
from tqdm import tqdm

from collections import defaultdict

from utils.dataset import get_all_datasets_names, load_dataset

from pathlib import Path

def get_info(dataset_name):
    dataset = load_dataset(dataset_name)
    dataset_rows = dataset.iloc[:,1:].to_numpy()
    length_ts = dataset_rows.shape[1]
    nr_instances = dataset_rows.shape[0]

    labels = dataset.iloc[:,0].to_numpy()
    count_labels = len(np.unique(labels))

    return dataset_name, nr_instances, length_ts, nr_instances, count_labels

def info_all_data():
    all_datasets = get_all_datasets_names()
    dataset_info = defaultdict(lambda : [])
    progress_bar = tqdm(all_datasets, desc="Processing datasets")
    for dataset_name in progress_bar:
        progress_bar.set_postfix({"Current":f"Processing {dataset_name}"})
        dataset_name, nr_instances, length_ts, nr_instances, count_labels = get_info(dataset_name)
        dataset_info["dataset_name"].append(dataset_name)
        dataset_info["num_rows"].append(nr_instances)
        dataset_info["ts_length"].append(length_ts)
        dataset_info["nr_instances"].append(nr_instances)
        dataset_info["count_labels"].append(count_labels)


    df = pd.DataFrame().from_dict(dataset_info)
    path = "UCRArchive_2018"
    file = "all_data_info.csv"
    df.to_csv(Path(path +"/" +file),index=False)

def select_a_subset_representatives_data():
    path = "UCRArchive_2018"
    file = "all_data_info.csv"
    df = pd.read_csv(Path(path+"/"+file))
    min_length = df.loc[df["ts_length"].idxmin(), "dataset_name"]
    #max_length = df.loc[df["ts_length"].idxmax(), "dataset_name"]
    simplest = ["Chinatown", "ItalyPowerDemand"] # Short and binary
    #min_classes = df.loc[df["count_labels"].idxmin(), "dataset_name"]
    #max_classes = df.loc[df["count_labels"].idxmax(), "dataset_name"]

    max_train_size = df.loc[df["num_rows"].idxmax(), "num_rows"]


    often_used = ["ECG200","GunPoint","Coffee"]


    all_dataset = [min_length, *simplest, *often_used, max_train_size]
    all_dataset = np.unique(np.array(all_dataset))

    path = "UCRArchive_2018"
    file = "selected_datasets.csv"
    df_selected = df[df["dataset_name"].isin(all_dataset)]

    df_selected.to_csv(Path(path + "/" + file), index=False)
if __name__ == "__main__":
    info_all_data()
    select_a_subset_representatives_data()