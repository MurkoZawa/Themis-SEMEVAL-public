from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import json 
import torch
from PIL import Image
import os

def clean_text(text):
        if text == "" or text == None:
            return "Blank"
        # Remove escape sequences and replace "\\n" with a space
        cleaned_text = re.sub(r'\\n', ' ', text)
        # Remove any other special characters or patterns as needed
        cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', cleaned_text)
        return cleaned_text

class Fakeddit_Dataset(Dataset):
        def __init__(self, annotations, img_dir, n_tokens, preprocessor=None, tokenizer=None):
            img_labels = []
            for annotation in annotations:
                img_labels.append([annotation["id"] + ".jpg", annotation["2_way_label"], annotation["clean_title"]])
            self.img_labels = pd.DataFrame(img_labels, columns=["id", "2_way_label", "clean_title"])
            self.img_dir = img_dir
            self.imgs_path = self.img_labels.iloc[:, 0]
            self.texts = self.img_labels.iloc[:, 2]
            self.texts = [clean_text(text) for text in self.texts]
            self.n_tokens = n_tokens

            self.preprocessor = preprocessor
            self.tokenizer = tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
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
                text = self.tokenizer(text, 
                                      return_tensors="pt",
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=False,
                                      max_length=self.n_tokens)
            if self.preprocessor:
                image = self.preprocessor(images=image, return_tensors="pt")
            
            return image, label, text

# Funzione per caricare i dati dal file TSV
def fakeddit_load_annotations_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
        annotations = df.to_dict(orient='records')
    except Exception as e:
        print(f"Errore nel caricare il file TSV: {e}")
        annotations = []
    return annotations

# Funzione per restituire un dataset caricato
def get_dataset(Dataset, n_tokens, processor, tokenizer, ann_path, img_dir):
     # Caricamento dei dataset
    annotations_train = fakeddit_load_annotations_file(ann_path)
    dataset = Dataset(
        annotations=annotations_train,
        img_dir=img_dir,
        n_tokens=n_tokens,
        preprocessor=processor,
        tokenizer=tokenizer
    )

    return dataset

if __name__ == '__main__':
     annotations_train = fakeddit_load_annotations_file("Fakeddit/annotations/subtask2b/train.tsv")
     print(annotations_train[0])