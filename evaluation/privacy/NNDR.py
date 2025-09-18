import tqdm

from Generators.train_rnn_vae import RNN_VAE
from Generators.train_timeGAN import TimeGAN
from Generators.train_cBetaVAE import cBetaVAE
from Generators.train_CGAN import CGAN
from TSC.miniROCKET import MiniROCKET
from TSC.multiROCKET import MultiROCKET
from TSC.TSForest import TSForest

from sklearn.metrics import accuracy_score, f1_score,balanced_accuracy_score

from matplotlib import pyplot as plt
import os
import numpy as np
from sktime.distances import dtw_distance
import json
from tqdm import tqdm
from collections import defaultdict


def find_nndr(instance, train_x):
    min_distance = float("inf")
    second_min_distance = float("inf")
    for x_instance in train_x:
        curr_dist = dtw_distance(x=instance, y=x_instance)

        if curr_dist < min_distance:
            second_min_distance = min_distance
            min_distance = curr_dist
            continue

        elif curr_dist < second_min_distance:
            second_min_distance = curr_dist
            continue

    return min_distance/second_min_distance

def evaluate_privacy(synthetic_train_x, real_train_x):
    """Evaluate synthetic data against real data"""
    nndr_scores = []
    idx_of_list = [i for i in range(len(synthetic_train_x))]
    selected_idx = np.random.choice(idx_of_list,size=20,replace=False)
    selected_synth_data = synthetic_train_x[selected_idx]
    for synth_x in tqdm(selected_synth_data):
        nndr_score = find_nndr(synth_x,real_train_x)
        nndr_scores.append(nndr_score)

    min_nndr = float(np.min(nndr_scores))
    avg_nndr = float(np.mean(nndr_scores))
    return min_nndr, avg_nndr



def main():
    from utils.dataset import get_selected_dataset_names, load_dataset
    from utils.save import load_synthetic_data

    generators = [RNN_VAE(), TimeGAN(), CGAN(), cBetaVAE()]
    dataset_names = get_selected_dataset_names()
    synth_epoch = 50
    for dataset_name in dataset_names:
        print(f"Evaluating {dataset_name}...")
        train_datasets = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
        train_data = train_datasets.iloc[:,1:].to_numpy()


        def my_dict():
            return defaultdict(my_dict)
        scores_gen_acc = my_dict()

        detailed_scores = my_dict()
        for generator in generators:
            print(f"Evaluating {generator.get_name()}...")
            synthetic_data = load_synthetic_data(dataset_name=dataset_name, generator_name=generator.get_name(), epochs=synth_epoch)

            all_synth_data = []
            all_synth_labels = []
            for label in synthetic_data:
                for row in synthetic_data[label]:
                    all_synth_data.append(row.flatten())
                    all_synth_labels.append(float(label))

            all_synth_data = np.array(all_synth_data)
            min_overall, avg_drc = evaluate_privacy(synthetic_train_x=all_synth_data, real_train_x=train_data)
            scores_gen_acc["MIN"][f"SYNTHETIC_{generator.get_name()}"] = min_overall
            scores_gen_acc["AVG"][f"SYNTHETIC_{generator.get_name()}"] = avg_drc

            detailed_scores[generator.get_name()]["MIN"] = min_overall
            detailed_scores[generator.get_name()]["AVG"] = avg_drc

        # Create Results directory if it doesn't exist
        results_dir = f"Results/PRIVACY/NNDR/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)

        # Save results to JSON
        results_path = os.path.join(results_dir, f"results_{dataset_name}_{synth_epoch}.json")
        with open(results_path, 'w') as f:
            json.dump(scores_gen_acc, f, indent=4)

        results_path_detail  = os.path.join(results_dir, f"detailed_results_{dataset_name}_{synth_epoch}.json")
        with open(results_path_detail, 'w') as f:
            json.dump(detailed_scores, f, indent=4)

        print(f"\nResults saved to {results_path}")

        # MIN PRIVACY EVAL
        print(f"Plotting MIN PRIVACY...")
        # Plot the results for comparison
        plt.figure(figsize=(10, 6))
        plt.bar(scores_gen_acc["MIN"].keys(), scores_gen_acc["MIN"].values())
        plt.ylabel("MIN NNDR")
        plt.ylim([0, 1])
        plt.title(fr"{dataset_name} comparing MIN DISTANCE$\uparrow$ across generators at {synth_epoch} epochs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"min_privacy_comparison_{dataset_name}_{synth_epoch}.png")
        plt.savefig(plot_path)
        plt.show()

        # avg PRIVACY EVAL
        print(f"Plotting AVG PRIVACY...")
        # Plot the results for comparison
        plt.figure(figsize=(10, 6))
        plt.bar(scores_gen_acc["AVG"].keys(), scores_gen_acc["AVG"].values())
        plt.ylabel("AVG NNDR")
        plt.ylim((0,1))
        plt.title(fr"{dataset_name} comparing AVG DISTANCE$\uparrow$ across generators at {synth_epoch} epochs ")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"avg_privacy_comparison_{dataset_name}_{synth_epoch}.png")
        plt.savefig(plot_path)
        plt.show()






if __name__ == "__main__":
    main()





