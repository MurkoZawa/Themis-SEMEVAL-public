import pandas as pd
import json

def count_labels_from_tsv(file_path, label_column):
    """
    Conta i campioni per ciascun valore di etichetta in un file TSV.

    :param file_path: Percorso del file TSV.
    :param label_column: Nome della colonna contenente le etichette.
    :return: Dizionario con il conteggio delle etichette.
    """
    df = pd.read_csv(file_path, sep='\t')
    counts = df[label_column].value_counts().to_dict()
    return counts

def count_labels_from_jsonl(file_path, label_key):
    """
    Conta i campioni per ciascun valore di etichetta in un file JSONL.

    :param file_path: Percorso del file JSONL.
    :param label_key: Chiave dell'etichetta nel JSON.
    :return: Dizionario con il conteggio delle etichette.
    """
    counts = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                annotation = json.loads(line.strip())  # Decodifica ogni riga come JSON
                label = annotation.get(label_key)
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1
            except json.JSONDecodeError as e:
                print(f"Errore nel decodificare la riga in {file_path}: {e}")
    return counts

# Percorsi dei file TSV
tsv_file_paths = {
    'train': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\Fakeddit\annotations\train.tsv",
    'val': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\Fakeddit\annotations\val.tsv",
    'test': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\Fakeddit\annotations\test.tsv"
}

# Percorsi dei file JSONL
jsonl_file_paths = {
    'train': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\annotations\subtask2b\train.jsonl",
    'val': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\annotations\subtask2b\val.jsonl",
    'test': r"C:\Users\utente\Desktop\Università\Themis-SEMEVAL-public\Themis-trial\annotations\subtask2b\test.jsonl"
}

# Conteggio delle etichette nei file TSV
print("Conteggio delle etichette nei file TSV:")
for split, file_path in tsv_file_paths.items():
    tsv_counts = count_labels_from_tsv(file_path, '2_way_label')
    print(f"{split.capitalize()} - Numero di campioni per etichetta: {tsv_counts}")

# Conteggio delle etichette nei file JSONL
print("\nConteggio delle etichette nei file JSONL:")
for split, file_path in jsonl_file_paths.items():
    jsonl_counts = count_labels_from_jsonl(file_path, 'label')
    print(f"{split.capitalize()} - Numero di campioni per etichetta: {jsonl_counts}")
