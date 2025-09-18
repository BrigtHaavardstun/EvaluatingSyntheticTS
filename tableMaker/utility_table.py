from Generators.train_rnn_vae import RNN_VAE
from Generators.train_timeGAN import TimeGAN
from Generators.train_cBetaVAE import cBetaVAE
from Generators.train_CGAN import CGAN
from utils.dataset import get_selected_dataset_names, load_dataset
from utils.save import load_synthetic_data

import numpy as np

import json
def get_max_accuracy_over_all(dataset_name, synth_epochs):
    generators = [RNN_VAE(), TimeGAN(), CGAN(), cBetaVAE()]
    generators_names = [gen.get_name() for gen in generators]
    tsc_models  = ["TSForest", "MiniROCKET", "MultiROCKET"]
    max_over_all = float("-inf")
    for generator_name in generators_names:
        for tsc_model in tsc_models:
            max_over_all = max(max_over_all, get_accuracy(generator_name, dataset_name, tsc_model, synth_epochs))
    return max_over_all

def get_accuracy(generator_name, dataset_name, model, synth_epochs):
    path = f"Results/UTILITY/{dataset_name}/detailed_results_{dataset_name}_{synth_epochs}.json"
    all_info = json.load(open(path))
    accuracy = all_info[generator_name]["SYNTHETIC" if generator_name != "REAL" else "REAL"][model]["Accuracy"]

    return accuracy

#def get_accuracy(generator_name, dataset_name, synth_epochs):
#    return get_metric(generator_name, dataset_name, "Accuracy", synth_epochs)

#def get_f1_score(generator_name, dataset_name, synth_epochs):
#    return get_metric(generator_name, dataset_name, "F1-Score", synth_epochs)

#def get_auc(generator_name, dataset_name, synth_epochs):
#    return get_metric(generator_name, dataset_name, "AUC", synth_epochs)

def make_table():
    generators = [RNN_VAE(), TimeGAN(), CGAN(), cBetaVAE()]
    generators_names = ["REAL"] + [gen.get_name() for gen in generators]
    dataset_names = get_selected_dataset_names()

    synth_epochs = 50
    latex_text = ""

    beginning = "\\begin{table}[!ht]\n\\centering\n\\label{table:utility}\n"
    latex_text += beginning

    models = ["", "MiniRocket", "MultiRocket"]

    table = ("\\begin{tabular}{|l|l|c|c|c|c|}\n\\hline"
             "\n\\textbf{Dataset} & \\textbf{Model} & \\textbf{TSForest} $\\uparrow$ & \\textbf{MiniROCKET} $\\uparrow$ & \\textbf{MultiROCKET} $\\uparrow$ & \\textbf{avg} \\\\\n")
    table += "\\hline\n"

    latex_text += table
    for dataset_name in dataset_names:
        curr_dataset = "\\multirow{5}*{"+dataset_name+"}\n"
        max_accuracy_over_all = get_max_accuracy_over_all(dataset_name, synth_epochs)
        for generator_name in generators_names:
            if generator_name == RNN_VAE().get_name():
                generator_row = f"& RNN\_VAE "
            elif generator_name == "REAL":
                generator_row = "& \\textit{Baseline}"
            else:
                generator_row = f"& {generator_name}"

            generator_ts_forest = get_accuracy(generator_name=generator_name, dataset_name=dataset_name,model="TSForest",synth_epochs=synth_epochs )
            generator_mini_rocket = get_accuracy(generator_name=generator_name, dataset_name=dataset_name,model="MiniROCKET",synth_epochs=synth_epochs )
            generator_multi_rocket = get_accuracy(generator_name=generator_name, dataset_name=dataset_name,model="MultiROCKET",synth_epochs=synth_epochs )

            avg_result =round((generator_ts_forest + generator_mini_rocket + generator_multi_rocket)/3,2)

            if max_accuracy_over_all in [generator_ts_forest, generator_mini_rocket, generator_multi_rocket]:
                if generator_name == RNN_VAE().get_name():
                    generator_row = "& \\textbf{RNN\_VAE}"
                else:
                    generator_row = "& \\textbf{" + generator_name +"}"

            ts_forest_text = str(round(generator_ts_forest,2))
            if max_accuracy_over_all == generator_ts_forest:
                ts_forest_text ="\\textbf{" + ts_forest_text  + "}"

            mini_rocket_text = str(round(generator_mini_rocket,2))
            if max_accuracy_over_all == generator_mini_rocket:
                mini_rocket_text = "\\textbf{" + mini_rocket_text + "}"

            multi_rocket_text = str(round(generator_multi_rocket,2))
            if max_accuracy_over_all == generator_multi_rocket:
                multi_rocket_text = "\\textbf{" + multi_rocket_text + "}"


            #if min_ims(generator_name, dataset_name, synth_epochs):
            #    generator_ims = "\\textbf{" + str(generator_ims) + "}"
            #if max_dcr(generator_name, dataset_name, synth_epochs):
            #    generator_dcr = "\\textbf{" + str(generator_dcr) + "}"
            #if max_nnda(generator_name, dataset_name, synth_epochs):
            #    generator_nnda = "\\textbf{" + str(generator_nnda) + "}"

            generator_row += f" & {ts_forest_text} & {mini_rocket_text} & { multi_rocket_text} & {avg_result}\\\\\n"

            curr_dataset += generator_row

        curr_dataset += "\\hline\n"
        latex_text += curr_dataset


    end_table = ""
    end_table += ("\\end{tabular}\n"
                  "\\caption{Utility scores for all models. In addition to the original generators, we have added a row containing the Baseline results when training on the original train data. We observe that the generators performance very greatly from dataset to dataset, and the generator with greatest utility seem to heavily depend on the dataset at hand.}\n"
                  "\\end{table}\n")


    latex_text += end_table
    print(latex_text)

if __name__ == "__main__":
    make_table()



