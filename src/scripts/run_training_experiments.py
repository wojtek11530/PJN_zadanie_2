import logging
import os
import sys

from src.settings import DATA_FOLDER, PROJECT_FOLDER

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

dataset_dir = os.path.relpath(os.path.join(DATA_FOLDER, 'dataset_conll'), start=PROJECT_FOLDER)
we_models_dir = os.path.relpath(os.path.join(DATA_FOLDER, 'models'), start=PROJECT_FOLDER)

REP_NUM = 1

word_embedding_models = [
    (os.path.join(we_models_dir, 'word2vec_train_base_cbow.model'), 'word2vec', False),
    (os.path.join(we_models_dir, 'word2vec_train_clean_cbow.model'), 'word2vec', True),
    (os.path.join(we_models_dir, 'word2vec_wiki_base_cbow.model'), 'word2vec', False),
    (os.path.join(we_models_dir, 'word2vec_wiki_clean_cbow.model'), 'word2vec', True),
    (os.path.join(we_models_dir, 'model_train_base_cbow.bin'), 'fasttext', False),
    (os.path.join(we_models_dir, 'model_train_clean_cbow.bin'), 'fasttext', True),
    (os.path.join(we_models_dir, 'model_wiki_base_cbow.bin'), 'fasttext', False),
    (os.path.join(we_models_dir, 'model_wiki_clean_cbow.bin'), 'fasttext', True),
    (os.path.join(we_models_dir, 'model_train_base_skip.bin'), 'fasttext', False),
    (os.path.join(we_models_dir, 'model_train_clean_skip.bin'), 'fasttext', True),
    (os.path.join(we_models_dir, 'model_wiki_base_skip.bin'), 'fasttext', False),
    (os.path.join(we_models_dir, 'model_wiki_clean_skip.bin'), 'fasttext', True)
]


def main():
    os.chdir(PROJECT_FOLDER)
    for we_model_dir, we_model_type, preprocessing in word_embedding_models:
        cmd = 'python -m src.scripts.train_model '
        options = [
            '--data_dir', dataset_dir,
            '--word_embedding_model_dir', we_model_dir,
            '--word_embedding_type', we_model_type,
            '--rep_num', str(REP_NUM),
            '--eval'
        ]
        cmd += ' '.join(options)
        if preprocessing:
            cmd += ' --preprocess_text'

        logger.info(f"Training for {os.path.basename(we_model_dir)}")
        run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
