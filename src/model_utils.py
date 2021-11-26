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
from tqdm.auto import tqdm

from src.dataset import TextDataModule, TextDataset
from src.mlp import MLPClassifier
from src.settings import MODELS_FOLDER
from src.utils import dictionary_to_json, is_folder_empty
from src.word_embedder import FasttextWordEmbedder, Word2VecWordEmbedder, TransformersWordEmbedder

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(args):
    if args.word_embedding_type == 'fasttext':
        word_embedder = FasttextWordEmbedder(args.word_embedding_model_dir)
        if 'cbow' in args.word_embedding_model_dir:
            args.word_embedding_type = args.word_embedding_type + '_cbow'
        else:
            args.word_embedding_type = args.word_embedding_type + '_skipgram'
    elif args.word_embedding_type == 'word2vec':
        word_embedder = Word2VecWordEmbedder(args.word_embedding_model_dir)
        if 'cbow' in args.word_embedding_model_dir:
            args.word_embedding_type = args.word_embedding_type + '_cbow'
        else:
            args.word_embedding_type = args.word_embedding_type + '_skipgram'
    elif args.word_embedding_type == 'transformers':
        word_embedder = TransformersWordEmbedder(args.word_embedding_model_dir)
    else:
        raise ValueError(f'Incorrect word embedding model type for: {args.word_embedding_type}')

    datamodule = TextDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        word_embedder=word_embedder,
        avg_embedding=True,
        preprocess_text=args.preprocess_text
    )

    rep_num = args.rep_num
    evaluate = args.eval

    dict_hyperparameters = vars(args)
    dict_hyperparameters.pop('rep_num')
    dict_hyperparameters.pop('eval')

    for i in range(rep_num):
        save_model_dir = manage_output_dir(
            model_name='MLP',
            word_embedding_model_type=args.word_embedding_type,
            word_embedding_model_dir=args.word_embedding_model_dir
        )

        dictionary_to_json(dict_hyperparameters, os.path.join(save_model_dir, 'hp.json'))
        tensorboard_logger = TensorBoardLogger(name='tensorboard_logs', save_dir=save_model_dir,
                                               default_hp_metric=False)

        trainer = pl.Trainer(
            logger=tensorboard_logger,
            max_epochs=args.epochs,
            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=6, verbose=True)],
            gpus=1 if torch.cuda.is_available() else None
        )

        model = MLPClassifier(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout
        )

        trainer.fit(model, datamodule)
        trainer.save_checkpoint(filepath=os.path.join(save_model_dir, 'model.chkpt'))

        if evaluate:
            print('Run model evaluation')
            evaluate_model(save_model_dir, args.data_dir, args.preprocess_text)


def evaluate_model(model_dir: str, data_dir: str, preprocess_text: bool) -> None:
    with open(os.path.join(model_dir, 'hp.json')) as json_file:
        hyperparams = json.load(json_file)

    model = MLPClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(model_dir, 'model.chkpt'),
        input_size=hyperparams['input_size'],
        hidden_size=hyperparams['hidden_size'],
        output_size=hyperparams['output_size'],
        learning_rate=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay'],
        dropout=hyperparams['dropout']
    )

    if hyperparams['word_embedding_type'] in ('fasttext', 'fasttext_cbow', 'fasttext_skipgram'):
        word_embedder = FasttextWordEmbedder(hyperparams['word_embedding_model_dir'])
    elif hyperparams['word_embedding_type'] in ('word2vec', 'word2vec_cbow', 'word2vec_skipgram'):
        word_embedder = Word2VecWordEmbedder(hyperparams['word_embedding_model_dir'])
    elif hyperparams['word_embedding_type'] == 'transformers':
        word_embedder = TransformersWordEmbedder(hyperparams['word_embedding_model_dir'])
    else:
        raise ValueError(f"Incorrect word embedding model type for: {hyperparams['word_embedding_type']}")

    dataset = TextDataset(
        filepath=os.path.join(data_dir, 'hotels.sentence.test.txt'),
        word_embedder=word_embedder,
        avg_embedding=True,
        preprocess_text=preprocess_text
    )
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


def test_model(model: torch.nn.Module, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    model = model.eval()
    predictions: List[torch.Tensor] = []
    real_values: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y_labels = batch
            logits = model(x)
            _, y_hat = torch.max(logits, dim=1)

            predictions.extend(y_hat)
            real_values.extend(y_labels)

    predictions_tensor = torch.stack(predictions).cpu()
    real_values_tensor = torch.stack(real_values).cpu()
    return predictions_tensor, real_values_tensor


def get_confusion_matrix_plot(conf_matrix: pd.DataFrame) -> Tuple[plt.figure, plt.Axes]:
    fig, ax = plt.subplots()
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                       annot_kws={"fontsize": 18}, square=True, ax=ax)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, fontsize=18)
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, fontsize=18)
    ax.set_ylabel('Rzecziwista klasa', fontsize=18)
    ax.set_xlabel('Predykowana klasa', fontsize=18)
    fig.tight_layout()
    return fig, ax


def manage_output_dir(model_name: str, word_embedding_model_type: str, word_embedding_model_dir: str) -> str:
    word_embedding_model_name = os.path.basename(word_embedding_model_dir)
    output_dir = os.path.join(MODELS_FOLDER,
                              model_name + '-' + word_embedding_model_type + '-' + word_embedding_model_name)
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
