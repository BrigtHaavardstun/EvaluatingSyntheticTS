import pandas as pd
from sdv.metadata import Metadata
import os
from pathlib import Path
def get_all_datasets_names():
    datasets_folder = "UCRArchive_2018"
    dataset_names = [item for item in os.listdir(datasets_folder)
                     if os.path.isdir(os.path.join(datasets_folder, item))]
    return dataset_names

def get_selected_dataset_names():
    df = pd.read_csv(Path("UCRArchive_2018","selected_datasets.csv"))
    return df.loc[:,"dataset_name"]

def load_data_by_class(dataset_name, dataset_type="TRAIN"):
    data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")

    grouped_by_class = data.groupby(data.columns[0])

    return grouped_by_class

def load_dataset(dataset_name, dataset_type="TRAIN"):
    data = pd.read_csv(f'UCRArchive_2018/{dataset_name}/{dataset_name}_{dataset_type}.tsv', sep='\t', header=None)
    data.columns = data.columns.astype(str)
    return data

def load_labels(dataset_name, dataset_type="TRAIN"):
    data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
    data.columns = data.columns.astype(str)
    labels = data.iloc[:,0]
    return list(set(labels))

def get_metadata(dataset_name):
    data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
    # Create metadata
    metadata = Metadata()
    table_name = dataset_name
    metadata.add_table(table_name)

    metadata.add_column(table_name=table_name, column_name=data.columns[0], sdtype='categorical')

    for col in data.columns[1:]:
        metadata.add_column(table_name=table_name, column_name=col, sdtype='numerical')

    metadata.validate()
    return metadata

def get_metadata_no_label(dataset_name:str):
    data = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
    # Create metadata
    metadata = Metadata()
    table_name = dataset_name
    metadata.add_table(table_name)

    for col in data.columns[1:]:
        metadata.add_column(table_name=table_name, column_name=col, sdtype='numerical')

    metadata.validate()
    return metadata

if __name__ == "__main__":
    print(load_labels("Chinatown"))
