import pandas as pd
import sdv
from sdv.single_table import CopulaGANSynthesizer, TVAESynthesizer, CTGANSynthesizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import os
import matplotlib.pyplot as plt
import json

def load_dataset(dataset_name, dataset_type="TRAIN"):
    """Loads a UCR dataset."""
    path = f'UCRArchive_2018/{dataset_name}/{dataset_name}_{dataset_type}.tsv'
    data = pd.read_csv(path, sep='\t', header=None)
    y = data[0]
    X = data.drop(columns=0)
    return X, y

def main(dataset_name):
    """Evaluates synthesizers, plots the results, and saves them to a JSON file."""
    # Load the real data
    X_train, y_train = load_dataset(dataset_name, "TRAIN")
    X_test, y_test = load_dataset(dataset_name, "TEST")

    # Train a classifier on the real data to get a baseline score
    clf_real = RandomForestClassifier(random_state=42)
    clf_real.fit(X_train, y_train)
    y_pred_real = clf_real.predict(X_test)
    accuracy_real = accuracy_score(y_test, y_pred_real)

    results = {"Real Data": accuracy_real}

    # Find all synthesizer files in the current directory
    generator_folder = "generators"
    synthesizer_files = [os.path.join(generator_folder,f) for f in os.listdir(generator_folder) if f.endswith('.pkl')]

    for synthesizer_path in synthesizer_files:
        print(f"Evaluating {synthesizer_path}...")
        # Load the synthesizer
        if "copula" in synthesizer_path:
            synthesizer = CopulaGANSynthesizer.load(synthesizer_path)
        elif "vae" in synthesizer_path:
            synthesizer = TVAESynthesizer.load(synthesizer_path)
        elif "ctgan" in synthesizer_path:
            synthesizer = CTGANSynthesizer.load(synthesizer_path)
        else:
            raise ValueError("Unknown synthesizer type")


        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=len(X_train))
        y_synth = synthetic_data[0]
        x_synth = synthetic_data.drop(columns=0)

        # Train a classifier on the synthetic data
        clf_synth = RandomForestClassifier(random_state=42)
        clf_synth.fit(x_synth, y_synth)  # Use synthetic features with real labels
        y_pred_synth = clf_synth.predict(X_test)
        accuracy_synth = accuracy_score(y_test, y_pred_synth)
        
        # Store the result
        model_name = os.path.splitext(synthesizer_path)[0].replace('_', ' ').title()
        results[model_name] = accuracy_synth

    # Create Results directory if it doesn't exist
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Save results to JSON
    results_path = os.path.join(results_dir, f"results_{dataset_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {results_path}")

    # Plot the results for comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy Score")
    plt.title(f"Classifier Utility Comparison on {dataset_name} Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    

    
    print(f"Utility comparison plot saved to {plot_path}")
    for model, score in results.items():
        print(f"- {model}: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate synthetic data utility.")
    parser.add_argument(
        "--dataset_name", 
        default="Chinatown",
        help="Name of the UCR dataset to use for evaluation (default: ACSF1)"
    )
    args = parser.parse_args()
    main(args.dataset_name)
