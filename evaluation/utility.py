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

import json

def score_all_classifiers(classifiers, score_dict, test_x, test_y):
    """Score all classifiers on real and synthetic data"""
    for classifier in classifiers:
        y_pred_real = classifier.predict(test_x)
        score_dict[classifier.NAME] = {
            "Accuracy": accuracy_score(test_y, y_pred_real),
            "F1-Score": f1_score(test_y, y_pred_real, average="macro"),
            "AUC": balanced_accuracy_score(test_y, y_pred_real)
        }

def score_real_data(real_train_x, real_train_y,test_x,test_y, epochs):
    # First train classifiers on real data
    classifiers_real = [TSForest(), MultiROCKET(), MiniROCKET()]
    for classifier in classifiers_real:
        classifier.train(real_train_x, real_train_y,epochs=epochs)


    scores = {"REAL":{}}
    score_all_classifiers(classifiers_real, scores["REAL"], test_x, test_y)
    print(json.dumps(scores["REAL"], indent=4))
    return scores

def evaluate_synthetic_data(synthetic_train_x, synthetic_train_y, test_x,test_y):
    """Evaluate synthetic data against real data"""
    epochs = 100

    scores = {"SYNTHETIC":{}}
    # First train classifiers on real data
    classifiers_synthetic = [TSForest(), MultiROCKET(), MiniROCKET()]
    for classifier in classifiers_synthetic:
        classifier.train(synthetic_train_x, synthetic_train_y, epochs=epochs)

    score_all_classifiers(classifiers_synthetic, scores["SYNTHETIC"], test_x, test_y)
    print(json.dumps(scores["SYNTHETIC"], indent=4))
    return scores



def main():
    from utils.dataset import get_selected_dataset_names, load_dataset
    from utils.save import load_synthetic_data

    generators = [RNN_VAE(), TimeGAN(), CGAN(), cBetaVAE()]
    dataset_names = get_selected_dataset_names()
    synth_epoch = 50
    train_tsc_epochs = 100

    for dataset_name in dataset_names:
        print(f"Evaluating {dataset_name}...")
        train_datasets = load_dataset(dataset_name=dataset_name, dataset_type="TRAIN")
        train_data = train_datasets.iloc[:,1:].to_numpy()
        train_labels = train_datasets.iloc[:,0].to_numpy()

        test_dataset = load_dataset(dataset_name=dataset_name, dataset_type="TEST")
        test_data = test_dataset.iloc[:,1:].to_numpy()
        test_labels = test_dataset.iloc[:,0].to_numpy()

        scores_gen_acc = {}
        detailed_scores = {}
        score_real = score_real_data(train_data, train_labels, test_data, test_labels, train_tsc_epochs)
        real_accuracy_average = np.average([score_real["REAL"][classifier]["Accuracy"] for classifier in score_real["REAL"]])
        scores_gen_acc["REAL_avg"] = real_accuracy_average

        detailed_scores["REAL"] = score_real
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
            all_synth_labels = np.array(all_synth_labels)

            # Only sample same number as original train
            picked_indexes = np.random.choice(len(all_synth_data), len(train_data))
            selected_synth_data = all_synth_data[picked_indexes]
            selected_synth_labels = all_synth_labels[picked_indexes]



            scores_syth = evaluate_synthetic_data(synthetic_train_x=selected_synth_data, synthetic_train_y=selected_synth_labels,
                                                  test_x=test_data,test_y=test_labels)

            synth_accuracy_average = np.average([scores_syth["SYNTHETIC"][classifier]["Accuracy"] for classifier in scores_syth["SYNTHETIC"]])
            scores_gen_acc[f"SYNTHETIC_{generator.get_name()}"] = synth_accuracy_average
            detailed_scores[generator.get_name()] = scores_syth

        # Create Results directory if it doesn't exist
        results_dir = f"Results/UTILITY/{dataset_name}"
        os.makedirs(results_dir, exist_ok=True)

        # Save results to JSON
        results_path = os.path.join(results_dir, f"results_{dataset_name}_{synth_epoch}.json")
        with open(results_path, 'w') as f:
            json.dump(scores_gen_acc, f, indent=4)

        results_path_detail  = os.path.join(results_dir, f"detailed_results_{dataset_name}_{synth_epoch}.json")
        with open(results_path_detail, 'w') as f:
            json.dump(detailed_scores, f, indent=4)

        print(f"\nResults saved to {results_path}")
        # Plot the results for comparison
        plt.figure(figsize=(10, 6))
        plt.bar(scores_gen_acc.keys(), scores_gen_acc.values())
        plt.ylabel("Accuracy Score")
        plt.title(f"{dataset_name} comparing utility across generators at {synth_epoch} epochs")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"utility_comparison_{dataset_name}_{synth_epoch}.png")
        plt.savefig(plot_path)
        plt.show()




if __name__ == "__main__":
    main()





