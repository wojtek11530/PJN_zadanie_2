import json
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_label_encoder(labels: np.ndarray):
    unique_labels = np.sort(np.unique(labels))
    label_encoder = LabelEncoder().fit(unique_labels)
    return label_encoder


def dictionary_to_json(dictionary: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dictionary, f, indent=2)


def is_folder_empty(folder_name: str):
    if len([f for f in os.listdir(folder_name) if not f.startswith('.')]) == 0:
        return True
    else:
        return False
