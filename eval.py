import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from themis_model import get_Themis
import re

import warnings
warnings.filterwarnings(action="ignore")

from datasets import get_dataset, DGM4_Dataset, dgm4_load_annotations_file, MFD_Dataset, mfd_load_annotations_file, Fakeddit_Dataset, fakeddit_load_annotations_file

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--name_llm", type=str)
    parser.add_argument("--name_img_embed", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--merge_tokens", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--use_lora", type=bool)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--n_tokens", type=int, default=128)
    parser.add_argument("--set_params", type=bool, default=False)
    parser.add_argument("--save_preds", type=bool, default=False)

    args = parser.parse_args()
    name_llm = args.name_llm
    name_img_embed = args.name_img_embed
    batch_size = args.batch_size
    merge_tokens = args.merge_tokens if args.merge_tokens != 0 else None
    lora_alpha = args.lora_alpha
    lora_r = args.lora_r
    lora_dropout = args.lora_dropout
    use_lora = args.use_lora
    model_path = args.model_path
    n_tokens = args.n_tokens
    set_params = args.set_params
    save_preds = args.save_preds

    if set_params:
        p = model_path.split('\\')[-1].split('_')
        lora_alpha = int(p[2])
        lora_r = int(p[3])
        lora_dropout = float(p[4])
        use_lora = True if 'True' in p[5] else False 
    
    model_dir = ''
    for i in model_path.split('\\')[:-1]:
        model_dir += i + '\\'
   
    themis, tokenizer, processor = get_Themis(
        name_llm=name_llm,
        name_img_embed=name_img_embed,
        use_lora=use_lora,
        is_pythia=True if "pythia" in name_llm else False,
        lora_alpha=lora_alpha,
        lora_r=lora_r,
        lora_dropout=lora_dropout,
        merge_tokens=merge_tokens
    )
    themis.to("cuda")

    base_dir = "MFD"
    dataset_test = get_dataset(MFD_Dataset, mfd_load_annotations_file, n_tokens, processor, tokenizer, 
                "MFD/test.tsv",
                "MFD/test")
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

    # Extract the predictions for the test set
    themis.load_state_dict(torch.load(model_path, map_location='cpu'))
    preds = []
    accumulated_labels = []
    running_loss = 0
    paths = []
    loss = nn.BCELoss()

    with torch.no_grad():
        for images, labels, text, batch_paths in tqdm(dataloader_test):
            images = images.to("cuda")
            text = text.to("cuda")
            labels = labels.to("cuda")

            outputs = themis(images, text)

            loss_test = loss(outputs.float(), labels.float().unsqueeze(1))
            running_loss += loss_test.item()
            preds.extend(outputs.cpu().detach().numpy())
            accumulated_labels.extend(labels.cpu().numpy())
            paths.extend(batch_paths)
        total_loss = running_loss / len(dataloader_test)
        preds_binary = [1 if i > 0.5 else 0 for i in preds]

        if save_preds:
            datas = [
                {"label": label, "pred": pred, "path": path}
                for label, pred, path in zip(accumulated_labels, preds_binary, paths)
            ]

            with open('pred.txt', "w") as output:
                output.write(str(datas))

        # Metrics calculations
        acc = accuracy_score(accumulated_labels, preds_binary)
        prec0 = precision_score(accumulated_labels, preds_binary, pos_label=0)
        rec0 = recall_score(accumulated_labels, preds_binary, pos_label=0)
        f10 = f1_score(accumulated_labels, preds_binary, pos_label=0)
        prec1 = precision_score(accumulated_labels, preds_binary, pos_label=1)
        rec1 = recall_score(accumulated_labels, preds_binary, pos_label=1)
        f11 = f1_score(accumulated_labels, preds_binary, pos_label=1)
        macro_f1 = f1_score(accumulated_labels, preds_binary, average='macro')
        conf_matr = confusion_matrix(accumulated_labels, preds_binary)

        # AUC calculation
        auc = roc_auc_score(accumulated_labels, preds)

        # ROC Curve and EER Calculation
        fpr, tpr, _ = roc_curve(accumulated_labels, preds, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        # Logging metrics
        print(f"Test loss: {total_loss}")
        print(f"Accuracy: {acc}")
        print(f"Label: 0 || Precision: {prec0} || Recall: {rec0} || F1: {f10}")
        print(f"Label: 1 || Precision: {prec1} || Recall: {rec1} || F1: {f11}")
        print(f"Macro-F1: {macro_f1}")
        print(f"AUC: {auc}")
        print(f"EER: {eer}")
        print(conf_matr)

        metrics = {
            "loss": total_loss,
            "accuracy": acc,
            "precision_0": prec0,
            "precision_1": prec1,
            "recall_0": rec0,
            "recall_1": rec1,
            "F1_0": f10,
            "F1_1": f11,
            "macro_F1": macro_f1,
            "AUC": auc,
            "EER": eer
        }

        # Save confusion matrix
        ax = plt.subplot()
        sns.heatmap(conf_matr, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['unreliable', 'reliable'])
        ax.yaxis.set_ticklabels(['unreliable', 'reliable'])
        plt.savefig(model_dir + 'conf_matr.png')

        # Save metrics as JSON
        with open(model_dir + 'metrics.json', "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        print("Done!")
