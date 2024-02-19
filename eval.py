import torch
import torch.nn as nn
import torch.optim as optim

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tqdm import tqdm
from themis_model import get_Themis
import re

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

    def clean_text(text):
        if text == "" or text == None:
            return "Blank"
        # Remove escape sequences and replace "\\n" with a space
        cleaned_text = re.sub(r'\\n', ' ', text)
        # Remove any other special characters or patterns as needed
        cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)
        return cleaned_text


    class EVALITA_Dataset(Dataset):
        def __init__(self, annotations_file, img_dir, preprocessor=None, tokenizer=None):
            file = open(annotations_file, "r", encoding="utf-8")
            annotations = json.load(file)

            #convert to pandas dataframe
            #if the label is "propagandistic" then 1 else 0
            img_labels = []
            for annotation in annotations:
                if annotation["label"] == "propagandistic":
                    img_labels.append([annotation["image"], 1, annotation["text"]])
                else:
                    img_labels.append([annotation["image"], 0, annotation["text"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["image", "label", "text"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(len(self.img_labels))
            print(len(self.imgs_path))
            print(len(self.texts))
        def __len__(self):
            return len(self.img_labels)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.imgs_path[idx])
            image = Image.open(img_path).convert("RGB")
            label = self.img_labels.iloc[idx, 1]
            text = self.texts[idx]
            if self.tokenizer:
                if text == "" or text == None:
                    text = "Blank"
                text = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, return_attention_mask=False, max_length=n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text
        

    class EVALITA_Test_Dataset(Dataset):
        def __init__(self, annotations_file, img_dir, preprocessor=None, tokenizer=None):
            file = open(annotations_file, "r", encoding="utf-8")
            annotations = json.load(file)

            #convert to pandas dataframe
            #if the label is "propagandistic" then 1 else 0
            img_labels = []
            for annotation in annotations:
                    img_labels.append([annotation["image"], annotation["id"], annotation["text"]])
            img_labels = pd.DataFrame(img_labels, columns=["image", "id", "text"])
            self.img_labels = img_labels
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.ids = self.img_labels.iloc[:, 1]
            self.texts = [clean_text(text) for text in self.texts]
            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(len(self.ids))
            print(len(self.imgs_path))
            print(len(self.texts))
        def __len__(self):
            return len(self.img_labels)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.imgs_path[idx])
            image = Image.open(img_path).convert("RGB")
            id = self.ids[idx]
            text = self.texts[idx]
            if self.tokenizer:
                if text == "" or text == None:
                    text = "Blank"
                text = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, return_attention_mask=False, max_length=128)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, id, text
        
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
    

    dataset_train = EVALITA_Dataset(
        annotations_file="annotations/subtask2b/train.json",
        img_dir="images/2b/train",
        preprocessor=processor,
        tokenizer=tokenizer
    )

    dataset_val = EVALITA_Dataset(
        annotations_file="annotations/subtask2b/val.json",
        img_dir="images/2b/val",
        preprocessor=processor,
        tokenizer=tokenizer
    )

    dataset_test = EVALITA_Test_Dataset(    
        annotations_file="annotations/subtask2b/dev_unlabeled.json",
        img_dir="images/2b/dev",
        preprocessor=processor,
        tokenizer=tokenizer
    )
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))


  
    #extract the predictions for the test set
    themis.load_state_dict(torch.load(model_path,map_location='cpu'))
    preds = []
    ids = []
    with torch.no_grad():
        for images, id, text in tqdm(dataloader_test):
            images = images.to("cuda")
            text = text.to("cuda")
            #print(text)
            outputs = themis(images, text)
            preds.extend(outputs.cpu().detach().numpy())
            ids.extend(id)
        preds = [1 if i > 0.5 else 0 for i in preds]
        #convert the predictions to the format required by the competition
        #"propagandistic" if the prediction is 1 else "non_propagandistic"
        preds = ["propagandistic" if i == 1 else "non_propagandistic" for i in preds]
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


