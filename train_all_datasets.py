from sdv.single_table import TVAESynthesizer

from Generators.train_rnn_vae import RNN_VAE
from Generators.train_timeGAN import TimeGAN
from Generators.train_cBetaVAE import cBetaVAE
from Generators.train_CGAN import CGAN
from Generators.base_generator import BaseGenerator

from utils.viz import load_and_visualize
from utils.dataset import get_selected_dataset_names, load_dataset
from tqdm import tqdm


def run_all_train(dataset_name:str, epochs:int=50, nr_samples:int=None):
    dataset = load_dataset(dataset_name, dataset_type="TRAIN")
    nr_samples = nr_samples if nr_samples is not None else len(dataset)
    generators: list[BaseGenerator] = [ CGAN()]#, RNN_VAE()]  # TimeGAN(), RNN_VAE(), cBetaVAE(),
    for generator in generators:
        print(f"Training {generator.name}...")
        generator.train(dataset_name=dataset_name, epochs=epochs, nr_samples=nr_samples)
        load_and_visualize(dataset_name=dataset_name,generator_name=generator.get_name(),epochs=epochs)


def run():
    dataset_names = ["ECG200"]# get_selected_dataset_names()
    epochs = 50
    progressbar = tqdm(dataset_names)
    for dataset_name in progressbar:
        progressbar.set_postfix({"Current": f"{dataset_name}"})
        run_all_train(dataset_name, epochs=epochs)

if __name__ == "__main__":
    run()