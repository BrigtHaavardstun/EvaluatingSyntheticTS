from Generators.train_rnn_vae import RNN_VAE
from Generators.train_timeGAN import TimeGAN
from Generators.train_cBetaVAE import cBetaVAE
from Generators.train_CGAN import CGAN
from utils.dataset import get_selected_dataset_names, load_dataset
from utils.save import load_synthetic_data

import json
def get_ims(generator_name, dataset_name, synth_epochs):
    path = f"Results/PRIVACY/IMS/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    ims_info = json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    ims = ims_info["SUM"][gen_prefix+generator_name]
    return ims
def min_ims(generator_name, dataset_name,synth_epochs):
    path = f"Results/PRIVACY/IMS/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    ims_info = json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    ims_curr = ims_info["SUM"][gen_prefix + generator_name]

    ims_max = min(ims_info["SUM"].values())

    return ims_curr == ims_max


def get_dcr(generator_name, dataset_name, synth_epochs):
    path = f"Results/PRIVACY/DCR/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    dcr_info =  json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    dcr = dcr_info["MIN"][gen_prefix+generator_name]
    return dcr

def max_dcr(generator_name, dataset_name,synth_epochs):
    path = f"Results/PRIVACY/DCR/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    dcr_info = json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    dcr_curr = dcr_info["MIN"][gen_prefix + generator_name]

    dcr_max = max(dcr_info["MIN"].values())
    return dcr_curr == dcr_max

def get_nnda(generator_name, dataset_name, synth_epochs):
    path = f"Results/PRIVACY/NNDr/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    nnda_info = json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    nnda = nnda_info["AVG"][gen_prefix+generator_name]
    return nnda

def max_nnda(generator_name, dataset_name,synth_epochs):
    path = f"Results/PRIVACY/NNDr/{dataset_name}/results_{dataset_name}_{synth_epochs}.json"
    nnda_info = json.load(open(path))
    gen_prefix = "SYNTHETIC_"
    nnda_curr = nnda_info["AVG"][gen_prefix + generator_name]

    nnda_max = max(nnda_info["AVG"].values())
    return nnda_curr == nnda_max

def make_table():
    generators = [RNN_VAE(), TimeGAN(), CGAN(), cBetaVAE()]
    generators_names = [gen.get_name() for gen in generators]
    dataset_names = get_selected_dataset_names()

    synth_epochs = 50
    latex_text = ""

    beginning = "\\begin{table}[!ht]\n\\centering\n\\label{table:privacy}\n"
    latex_text += beginning

    table = ("\\begin{tabular}{|l|l|c|c|c|}\n\\hline"
             "\n\\textbf{Dataset} & \\textbf{Generator} & \\textbf{IMS} $\\downarrow$ & \\textbf{min. DCR} $\\uparrow$ & \\textbf{avg. NNDR} $\\uparrow$ \\\\\n")
    table += "\\hline\n"

    latex_text += table
    for dataset_name in dataset_names:
        curr_dataset = "\\multirow{4}*{"+dataset_name+"} "
        for generator_name in generators_names:
            if generator_name == RNN_VAE().get_name():
                generator_row = f"& RNN\_VAE "
            else:
                generator_row = f"& {generator_name}"

            generator_ims = round(get_ims(generator_name, dataset_name,synth_epochs ),2)
            generator_dcr = round(get_dcr(generator_name, dataset_name, synth_epochs),2)
            generator_nnda = round(get_nnda(generator_name, dataset_name, synth_epochs),2)

            if min_ims(generator_name, dataset_name, synth_epochs):
                generator_ims = "\\textbf{" + str(generator_ims) + "}"
            if max_dcr(generator_name, dataset_name, synth_epochs):
                generator_dcr = "\\textbf{" + str(generator_dcr) + "}"
            if max_nnda(generator_name, dataset_name, synth_epochs):
                generator_nnda = "\\textbf{" + str(generator_nnda) + "}"

            generator_row += f" & {generator_ims} & {generator_dcr} & {generator_nnda} \\\\\n"

            curr_dataset += generator_row

        curr_dataset += "\\hline\n"
        latex_text += curr_dataset


    end_table = ""
    end_table += ("\\end{tabular}\n"
                  "\\caption{Privacy scores across datasets and models. We observe that IMS is always zero, where as DCR and NNDR differ more, where we mark the best performance with \\textbf{bold}. As DCR is a non normalized metric, comparison across datasets is not directly comparable. However, the ranking of generators are still comparable for all privacy metrics. }\n"
                  "\\end{table}\n")


    latex_text += end_table
    print(latex_text)

if __name__ == "__main__":
    make_table()



