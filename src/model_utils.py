import datetime
import json
import logging
import os
import sys
from typing import Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import TextDataModule, TextDataset
from src.mlp import MLPClassifier
from src.settings import DATA_FOLDER
from src.utils import dictionary_to_json, is_folder_empty

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(args):
    output_dir = manage_output_dir(model_name='MLP')
    dictionary_to_json(vars(args), os.path.join(output_dir, 'hp.json'))

    logger = TensorBoardLogger(name='tensorboard_logs', save_dir=output_dir, default_hp_metric=False)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=50,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True),
        gpus=1
    )

    model = MLPClassifier(args)
    datamodule = TextDataModule(args.data_dir, args.batch_size, True)

    trainer.fit(model, datamodule)

    trainer.save_checkpoint(filepath=os.path.join(output_dir, 'model.chkpt'))
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.bin'))


def evaluate_model(model_dir: str, data_dir: str) -> None:
    with open(os.path.join(model_dir, 'hp.json')) as json_file:
        hyperparams = json.load(json_file)

    model = MLPClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(model_dir, 'model.chkpt'),
        **hyperparams,
    )
    dataset = TextDataset(os.path.join(data_dir, 'test_set.csv'), avg_embedding=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    classes_names = dataset.label_encoder.classes_

    y_pred, y_true = test_model(model, dataloader=dataloader)

    print('\n\t**** Classification report ****\n')
    print(classification_report(y_true, y_pred, target_names=classes_names))

    report = classification_report(y_true, y_pred, target_names=classes_names, output_dict=True)
    dictionary_to_json(report, os.path.join(model_dir, "test_results.json"))

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes_names, columns=classes_names)
    fig, ax = get_confusion_matrix_plot(df_cm)
    fig.savefig(os.path.join(model_dir, 'confusion_matrix.pdf'), bbox_inches='tight')


def test_model(model, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
    model = model.eval()
    predictions: List[torch.Tensor] = []
    real_values: List[torch.Tensor] = []
    with torch.no_grad():
        total = len(dataloader.dataset)
        current = 0
        for batch in dataloader:
            x, y_labels = batch
            logits = model(x)
            _, y_hat = torch.max(logits, dim=1)

            predictions.extend(y_hat)
            real_values.extend(y_labels)
            current += len(y_hat)
            print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}: {current}/{total}")

    predictions_tensor = torch.stack(predictions).cpu()
    real_values_tensor = torch.stack(real_values).cpu()
    return predictions_tensor, real_values_tensor


def get_confusion_matrix_plot(conf_matrix: pd.DataFrame):
    fig, ax = plt.subplots()
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                       annot_kws={"fontsize": 18}, square=True, ax=ax)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, fontsize=18)
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, fontsize=18)
    ax.set_ylabel('Rzecziwista klasa', fontsize=18)
    ax.set_xlabel('Predykowana klasa', fontsize=18)
    fig.tight_layout()
    return fig, ax


def manage_output_dir(model_name: str) -> str:
    output_dir = os.path.join(DATA_FOLDER, model_name)
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
