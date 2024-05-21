import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

import torch , torch.nn as nn

BASE_DIR = "./"

REPORT_DIR = osp.join(BASE_DIR, "report")

if not osp.exists(REPORT_DIR): os.mkdir(REPORT_DIR)

# if not osp.exists(osp.join(REPORT_DIR, "train_1")): os.mkdir(osp.join(REPORT_DIR, "train_1"))



def plot_loss(dir: str, results: dict[list[float]]):

    loss = np.array(results["loss"])

    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(osp.join(dir, "loss_plot.png"))


def save_model_architecture(dir: str, model: nn.Module, hyparms: dict):

    with open(osp.join(dir, "train_details.txt"), "w") as f:
        f.write(" ".join(["="*10, "model architecture", "="*10, "\n\n"]))
        f.write(str(model))
        f.write("\n\n")
        f.write(" ".join(["="*10, "hyperparameter tuning", "="*8, "\n"]))
        f.write(str(hyparms))
    

def get_next_report_path() -> str:

    reports = os.listdir(REPORT_DIR)

    index = 0 

    reports_list = sorted(reports, key= lambda path: -int(path.split("_")[1]))

    index = int(reports_list[0].split("_")[1]) + 1

    next_report_path = osp.join(REPORT_DIR, f"train_{index}")
    
    if not osp.exists(next_report_path): os.mkdir(next_report_path)

    return next_report_path




