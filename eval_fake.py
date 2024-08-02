import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tqdm import tqdm
from themis_model import get_Themis
import re
import warnings
warnings.filterwarnings(action="ignore")

from datasets import get_dataset, Fakeddit_Dataset

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


    name_llm = parser.parse_args().name_llm
    name_img_embed = parser.parse_args().name_img_embed
    batch_size = parser.parse_args().batch_size
    merge_tokens = parser.parse_args().merge_tokens
    if merge_tokens == 0:
        merge_tokens = None
    lora_alpha = parser.parse_args().lora_alpha
    lora_r = parser.parse_args().lora_r
    lora_dropout = parser.parse_args().lora_dropout
    use_lora = parser.parse_args().use_lora
    model_path = parser.parse_args().model_path
    n_tokens = parser.parse_args().n_tokens

            
    themis, tokenizer, processor = get_Themis(
        name_llm = name_llm,
        name_img_embed = name_img_embed,
        use_lora = use_lora,
        is_pythia = True if "pythia" in name_llm else False,
        lora_alpha = lora_alpha,
        lora_r = lora_r,
        lora_dropout = lora_dropout,
        merge_tokens = merge_tokens
    )
    themis.to("cuda")

    dataset_name = "Fakeddit"

    dataset_test = get_dataset(Fakeddit_Dataset, n_tokens, processor, tokenizer, 
                f"{dataset_name}/annotations/test.tsv",
                f"{dataset_name}/images/test")
    
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))

    #extract the predictions for the test set
    themis.load_state_dict(torch.load(model_path,map_location='cpu'))
    preds = []
    accumulated_labels = []
    running_loss = 0
    loss = nn.BCELoss()
    with torch.no_grad():
        for images, labels, text in tqdm(dataloader_test):
            images = images.to("cuda")
            text = text.to("cuda")
            labels = labels.to("cuda")
            
            outputs = themis(images, text)
            loss_test = loss(outputs.float(), labels.float().unsqueeze(1))
            running_loss += loss_test.item()
            preds.extend(outputs.cpu().detach().numpy())
            accumulated_labels.extend(labels.cpu().numpy())
        
        total_loss = running_loss / len(dataloader_test)
        preds = [1 if i > 0.5 else 0 for i in preds]

        acc = accuracy_score(accumulated_labels, preds)
        prec = precision_score(accumulated_labels, preds)
        rec = recall_score(accumulated_labels, preds)
        f1 = f1_score(accumulated_labels, preds)
        conf_matr = confusion_matrix(accumulated_labels, preds)
        print(f"Test loss: {total_loss}")
        print(f"Accuracy: {acc} || Precision: {prec} || Recall: {rec} || F1: {f1}")
        print(conf_matr)
        exit()
        #save the predictions to a json file
        json_preds = []
        for id,label in zip(ids,preds):
            current={
                "id": id,
                "label": label
            }
            json_preds.append(current)
        #remove / or \ from the name of the model and the image embedder
        name_llm = name_llm.replace("/","_").replace("\\","_")
        name_img_embed = name_img_embed.replace("/","_").replace("\\","_")

        name_out = "outputs/predictions_"+name_llm+"_"+name_img_embed+"_"+str(batch_size)+"_"+str(merge_tokens)+"_"+str(lora_alpha)+"_"+str(lora_r)+"_"+str(lora_dropout)+"_"+str(use_lora)+".json"
        with open(name_out, "w", encoding="utf-8") as f:
            json.dump(json_preds, f, indent=4, ensure_ascii=False)
        print("Done!")


