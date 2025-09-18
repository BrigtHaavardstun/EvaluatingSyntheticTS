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
from sktime.distances import dtw_distance, euclidean_distance
import json
from tqdm import tqdm
from collections import defaultdict
def save_my_dict_to_json(my_dict, path):
    def get_dict_or_value(my_dict):
        if not hasattr(my_dict, "keys"):
            return my_dict
        else:
            curr_level = {}
            for key in my_dict.keys():
                curr_level[key] = get_dict_or_value(my_dict[key])
            if not list(my_dict.keys()):
                curr_level = my_dict
            return curr_level
    json_able_dict = get_dict_or_value(my_dict)
    with open(path, 'w') as f:
        json.dump(json_able_dict, f, indent=4)




def find_IMS(instance, train_x, error=None):
    if error is None:
        min_y = min(instance)
        max_y = max(instance)
        error = (max_y - min_y)*0.1
    has_match = 0
    error_float = error # floating error stuff
    for x_instance in train_x:
        curr_dist = euclidean_distance(x=instance, y=x_instance)
        if curr_dist <= error_float:
            has_match = 1
            break

    return has_match

def evaluate_privacy(synthetic_train_x, real_train_x):
    """Evaluate synthetic data against real data"""
    from sklearn.preprocessing import MinMaxScaler
    all_distances = []
    idx_of_list = [i for i in range(len(synthetic_train_x))]
    selected_idx = np.random.choice(idx_of_list,size=20,replace=False)
    selected_synth_data = synthetic_train_x[selected_idx]
    for synth_x in tqdm(selected_synth_data):
        minimum_dist = find_IMS(synth_x,real_train_x)
        all_distances.append(minimum_dist)
        all_distances.append(minimum_dist)

    sum_IMS_overall = sum(all_distances)
    avg_IMS = np.mean(all_distances)
    return sum_IMS_overall, avg_IMS



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
            for label in synthetic_data.keys():
                for row in synthetic_data[label]:
                    all_synth_data.append(row.flatten())
                    all_synth_labels.append(float(label))

            all_synth_data = np.array(all_synth_data)

            sum_IMS_overall, avg_IMS = evaluate_privacy(synthetic_train_x=all_synth_data, real_train_x=train_data)
            scores_gen_acc["SUM"][f"SYNTHETIC_{generator.get_name()}"] = float(sum_IMS_overall)
            scores_gen_acc["AVG"][f"SYNTHETIC_{generator.get_name()}"] = float(avg_IMS)

            detailed_scores[generator.get_name()]["SUM"] = float(sum_IMS_overall)
            detailed_scores[generator.get_name()]["AVG"] = float(avg_IMS)

        # Create Results directory if it doesn't exist
        results_dir = f"Results/PRIVACY/IMS/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)

        # Save results to JSON
        results_path = os.path.join(results_dir, f"results_{dataset_name}_{synth_epoch}.json")
        save_my_dict_to_json(scores_gen_acc, results_path)


        results_path_detail  = os.path.join(results_dir, f"detailed_results_{dataset_name}_{synth_epoch}.json")
        with open(results_path_detail, 'w') as f:
            json.dump(detailed_scores, f, indent=4)

        print(f"\nResults saved to {results_path}")

        # COUNT PRIVACY EVAL
        print(f"Plotting MIN PRIVACY...")
        # Plot the results for comparison
        plt.figure(figsize=(10, 6))
        plt.bar(scores_gen_acc["SUM"].keys(), scores_gen_acc["SUM"].values())
        plt.ylabel("Total IMS")
        plt.title(fr"{dataset_name} comparing TOTAL IMS$\downarrow$ across generators at {synth_epoch} epochs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"min_privacy_comparison_{dataset_name}_{synth_epoch}.png")
        plt.savefig(plot_path)
        plt.show()

        # avg PRIVACY EVAL
        print(f"Plotting AVG PRIVACY...")
        # Plot the results for comparison
        plt.figure(figsize=(10, 6))
        plt.bar(scores_gen_acc["AVG"].keys(), np.array(list(scores_gen_acc["AVG"].values()))*100)
        plt.ylabel("Percentage IMS")
        plt.title(fr"{dataset_name} comparing AVG DISTANCE$\downarrow$ across generators at {synth_epoch} epochs ")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"avg_privacy_comparison_{dataset_name}_{synth_epoch}.png")
        plt.savefig(plot_path)
        plt.show()






if __name__ == "__main__":
    main()





