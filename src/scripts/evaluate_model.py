import argparse
import logging
import sys

from src.model_utils import evaluate_model

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The direction of folder where model.chkpt and hp.json files are located")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The data dir with files hotels.sentence.train.txt, "
                             "hotels.sentence.dev.txt, hotels.sentence.test.txt")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    evaluate_model(args.model_dir, args.data_dir)


if __name__ == '__main__':
    main()
