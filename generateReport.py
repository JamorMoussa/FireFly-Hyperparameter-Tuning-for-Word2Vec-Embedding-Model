import os
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt

import torch , torch.nn as nn

from sklearn.decomposition import PCA

BASE_DIR = "./"

REPORT_DIR = osp.join(BASE_DIR, "report")

if not osp.exists(REPORT_DIR): os.mkdir(REPORT_DIR)


def plot_loss(dir: str, results: dict[list[float]]):

    loss = np.array(results["loss"])

    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(osp.join(dir, "loss_plot.png"))
    plt.clf() 


def save_model_architecture(dir: str, model: nn.Module, hyparms: dict):

    with open(osp.join(dir, "train_details.txt"), "w") as f:
        f.write(" ".join(["="*10, "model architecture", "="*10, "\n\n"]))
        f.write(str(model))
        f.write("\n\n")
        f.write(" ".join(["="*10, "hyperparameter tuning", "="*8, "\n"]))
        f.write(str(hyparms))

    torch.save(model.state_dict(), osp.join(dir, "CBoWModel.pt"))
    

def get_next_report_path() -> str:

    reports = os.listdir(REPORT_DIR)

    index = 0 

    try:
        reports_list = sorted(reports, key= lambda path: -int(path.split("_")[1]))

        index = int(reports_list[0].split("_")[1]) + 1

    except IndexError: pass
    except Exception as e:
        print(e) 
    
    next_report_path = osp.join(REPORT_DIR, f"train_{index}")

    if not osp.exists(next_report_path): os.mkdir(next_report_path)

    return next_report_path


def plot_word_embedding_in_2d_space(
    dir: str,
    model: nn.Module,
    vocab:  dict[str, int]
):

    pca = PCA(n_components=2)

    words = ["king", "man","woman", "prince", "princess", "queen"]

    stack = []

    for word in words:
        stack.append(model[0](torch.tensor(vocab[word])))

    array = pca.fit_transform(torch.stack(stack).detach().numpy())

    plt.scatter(array[:, 0], array[:, 1])

    for i, label in enumerate(words):
        plt.annotate(label, (array[i, 0], array[i, 1]))

    plt.savefig(osp.join(dir, "word_plot.png"))
    plt.clf() 



