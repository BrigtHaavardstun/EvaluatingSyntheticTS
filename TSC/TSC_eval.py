import numpy as np
import pandas as pd
from click import progressbar
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Import sktime components
from sktime.datasets import load_arrow_head, load_italy_power_demand
from sktime.datatypes._panel._convert import from_2d_array_to_nested

from TSC.BOSS import BOSS
from TSC.KNN_DTW import KNN_DTW
from TSC.miniROCKET import MiniROCKET
from TSC.multiROCKET import MultiROCKET
from TSC.TSForest import TSForest

def load_sample_data():
    """Load sample time series classification data"""
    # Load a sample dataset (Arrow Head dataset)
    X_train, y_train = load_arrow_head(split="train", return_X_y=True)
    X_test, y_test = load_arrow_head(split="test", return_X_y=True)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")

    return X_train, X_test, y_train, y_test


def evaluate_classifier_basic(classifier, X_train, X_test, y_train, y_test, classifier_name="Classifier"):
    """Basic evaluation of a sktime classifier"""
    print(f"\n{'=' * 50}")
    print(f"Evaluating {classifier_name}")
    print(f"{'=' * 50}")

    # Fit the classifier
    print("Training classifier...")
    classifier.train(X_train, y_train, epochs=10)

    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    return accuracy, y_pred, cm


def evaluate_classifier_cv(classifier, X_train, y_train, cv_folds=5, classifier_name="Classifier"):
    """Evaluate classifier using cross-validation"""
    print(f"\n{'=' * 50}")
    print(f"Cross-Validation Evaluation for {classifier_name}")
    print(f"{'=' * 50}")

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=skf, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return cv_scores


def plot_confusion_matrix(cm, class_names, classifier_name="Classifier"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_sample_series(X, y, n_samples=3):
    """Plot sample time series from the dataset"""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 8))

    for i in range(n_samples):
        if n_samples == 1:
            ax = axes
        else:
            ax = axes[i]

        # Plot the time series
        plt.plot(X[i],y[i],ax=ax, title=f'Sample {i + 1}, Class: {y[i]}')

    plt.tight_layout()
    plt.show()


def compare_multiple_classifiers(X_train, X_test, y_train, y_test, classifiers=None,dataset_name=None):
    """Compare multiple sktime classifiers"""
    print(f"\n{'=' * 60}")
    print("COMPARING MULTIPLE CLASSIFIERS")
    print(f"{'=' * 60}")

    # Define classifiers to compare



    results = {}

    for classifier in classifiers:
        print(f"\nTraining {classifier.get_name()}...")
        try:
            accuracy, y_pred, cm = evaluate_classifier_basic(
                classifier, X_train, X_test, y_train, y_test, classifier.get_name()
            )
            results[classifier.get_name()] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'confusion_matrix': cm
            }
        except Exception as e:
            print(f"Error training {classifier.get_name()}: {str(e)}")

    # Plot comparison
    if results:
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title(f'{dataset_name} - Classifier Comparison - Test Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    return results


def main(dataset_name):
    """Main evaluation pipeline"""
    print("Starting sktime Classifier Evaluation")

    # Load data
    from utils.dataset import load_dataset
    train_data = load_dataset(dataset_name, "TRAIN")
    test_data = load_dataset(dataset_name, "TEST")
    X_train = train_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    X_test = test_data.iloc[:, 1:]
    y_test = test_data.iloc[:, 0]

    X_train = from_2d_array_to_nested(X_train)  # Each row becomes a Series
    X_test = from_2d_array_to_nested(X_test)

    classifiers = [TSForest(), MultiROCKET(), MiniROCKET()]
    # Plot some sample series
    print("\nPlotting sample time series...")
    #plot_sample_series(X_train, y_train, n_samples=3)

    # Single classifier evaluation
    print("\n" + "=" * 60)
    print("SINGLE CLASSIFIER EVALUATION")
    print("=" * 60)


    # Multiple classifier comparison
    compare_multiple_classifiers(X_train, X_test, y_train, y_test, classifiers,dataset_name)

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")


def run():
    # Run the evaluation
    from utils.dataset import get_selected_dataset_names
    from tqdm import tqdm
    dataset_names = get_selected_dataset_names()
    progressbar = tqdm(dataset_names)
    for dataset_name in tqdm(dataset_names):
        progressbar.set_postfix({"Current": f"{dataset_name}"})
        main(dataset_name=dataset_name)
if __name__ == "__main__":
    run()
