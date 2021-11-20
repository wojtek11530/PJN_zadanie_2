import json
import os
from typing import Any, Dict

import pandas as pd

from src.settings import DATA_FOLDER, MODELS_FOLDER


def main():
    models_subdirectories = get_immediate_subdirectories(MODELS_FOLDER)

    data = list()
    for subdirectory in models_subdirectories:
        data_dict = gather_results(subdirectory)
        data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, 'hp.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)
    del data['data_dir']
    w_emb_model_name = os.path.basename(data['word_embedding_model_dir'])
    del data['word_embedding_model_dir']

    data['full_name'] = os.path.basename(model_dir)
    data['name'] = data['full_name'].split('-run-')[0]
    if 'wiki' in w_emb_model_name:
        data['source_corpus'] = 'wikipedia'
    elif 'train' in w_emb_model_name:
        data['source_corpus'] = 'tripadvisor'
    else:
        raise ValueError

    return data


if __name__ == '__main__':
    main()
