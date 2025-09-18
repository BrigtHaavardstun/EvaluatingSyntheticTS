from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from utils.save import load_synthetic_data
from utils.dataset import load_data_by_class,load_dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_and_visualize(dataset_name,generator_name,epochs):
    synth_data = load_synthetic_data(dataset_name=dataset_name,generator_name=generator_name,epochs=epochs)
    colors = plt.cm.tab10(range(len(synth_data.keys())))
    col = dict(zip(synth_data.keys(), colors))
    for label in synth_data:
        for i, row in enumerate(synth_data[label]):
            plt.plot(row, color=col[label],alpha=0.8, label=f"Class {label} synth" if i == 0 else "")

    grouped_data = load_data_by_class(dataset_name=dataset_name)
    for label, data in grouped_data:
        no_label = data.iloc[:10,1:]
        for i, row in no_label.iterrows():
            plt.plot(row, color="grey",alpha=0.2)

    plt.legend()
    plt.title(f"{generator_name} Synthesizer for {dataset_name}")
    plt.show()


def load_and_tsne(dataset_name, generator_name, epochs):
    tsne = TSNE(n_components=2, verbose=1, perplexity=15)
    synth_data = load_synthetic_data(dataset_name=dataset_name, generator_name=generator_name, epochs=epochs)
    synth_list = []
    labels = []

    for key in synth_data.keys():
        for row in synth_data[key]:
            synth_list.append(row.flatten())
            labels.append("Synth")

    org_data = load_dataset(dataset_name=dataset_name)
    org_list = org_data.iloc[:,1:].to_numpy()
    org_flatt_list =[]
    for row in org_list:
        org_flatt_list.append(row.flatten())
        labels.append("Org")

    all_data = np.concatenate((synth_list,org_flatt_list))

    tsne_results = tsne.fit_transform(all_data)
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset['label'] =  labels

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", len(set(labels))),
        data=df_subset,
        legend="full",
        alpha=0.9
    )

    plt.show()

if __name__ == "__main__":
    load_and_tsne(dataset_name="ItalyPowerDemand", generator_name="timeGAN", epochs=1000)