import numpy as np
from reader import read_data
import matplotlib.pyplot as plt
import csv
import os
from random import randint
from copy import deepcopy

DATA_FOLDER = r'/Users/juliavaghy/Desktop/0--data'
data_file = 'data1/nonoise.csv'

# init params
params = {}
params["nmf_type"] = "NMFD"
params["fixW"] = "adaptive"
params["beta"] = 0
params["addedCompW"] = 0
params["window"] = 512
params["hop"] = int(params["window"]/2)
params["noise"] = "None"
params["noise-lvl"] = 0

nmf_types = ["NMF", "NMFD"]
fixW_options = ["fixed", "semi", "adaptive"]
addedCompWs = [0, 1, 2, 3, 4, 5]
noise_lvls = [0, 1, 2, 3]

while os.path.exists(data_file):
    data_file = 'data/' + str(randint(0, 100)) + '.csv'
print(f"Writing results in file {data_file}")

with open(data_file, 'w') as csv_file:
    header = [key for key, value in params.items() if key != "window" and key != "hop"] + ["Sample", "F", "P", "R"]
    writer = csv.writer(csv_file)
    writer.writerow(header)

    for noise_lvl in noise_lvls:
        if noise_lvl == 0:
            noises = ["None"]
        else:
            noise_dir = "background-loud"
            noises = ["airplane", "chatter", "ambient"]
            if noise_lvl == 1:
                noise_dir = "background"
            if noise_lvl == 3:
                noises =[ "mix"]

        params["noise-lvl"] = noise_lvl
        
        for fixW_option in fixW_options:
            params["fixW"] = fixW_option
            betas = [1, 2, 3, 4, 5, 6]
            if fixW_option == "adaptive":
                betas = [0]
            elif fixW_option == "fixed":
                betas = [float('inf')]
            for beta in betas:
                params["beta"] = beta
                for addedCompW in addedCompWs:
                    params["addedCompW"] = addedCompW
                    for noise in noises:
                        params["noise"] = noise
                        for nmf_type in nmf_types:
                            params["nmf_type"] = nmf_type
                            print(params)

                            samples = read_data(DATA_FOLDER, params)
                            precision = np.zeros(len(samples))
                            recall = np.zeros(len(samples))
                            f_measure = np.zeros(len(samples))

                            samples = read_data(DATA_FOLDER, params)
                            for sample_idx in range(len(samples)):
                                precision[sample_idx], recall[sample_idx], f_measure[sample_idx] = samples[sample_idx].evaluate()

                            print(np.mean(f_measure).round(2))

                            for idx in range(len(samples)):
                                param_values = [value for key, value in params.items() if key != "window" and key != "hop"] + [samples[idx].dir] + [f_measure[idx], precision[idx], recall[idx]]
                                print(param_values)
                                writer.writerow(param_values)