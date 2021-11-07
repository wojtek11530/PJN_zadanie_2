import argparse
import logging
import sys

from src.model_utils import train_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The data dir with files train_set.csv, dev_set.csv, test_set.csv")
    parser.add_argument("--input_size",
                        default=100,
                        type=int,
                        help="The dimension of input (i.e. word embeddings dimension)")
    parser.add_argument("--hidden_size",
                        default=32,
                        type=int,
                        help="The dimension of hidden layer")
    parser.add_argument("--output_size",
                        default=4,
                        type=int,
                        help="The output size of model, i.e. classes number")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="Probability of dropout")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    train_model(args)


if __name__ == '__main__':
    main()
