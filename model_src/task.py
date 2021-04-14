
import os
import sys
import logging
import tensorflow as tf
from tensorflow.python.client import device_lib
import argparse


try:
    from model_src import defaults, trainer, exporter
except:
    import defaults, trainer, exporter
    
dirname = os.path.dirname(__file__)
RAW_SCHEMA_DIR = os.path.join(dirname, 'raw_schema/schema.pbtxt')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-dir', 
        default=os.getenv("AIP_MODEL_DIR"), 
        type=str,
    )
    
    parser.add_argument(
        '--log-dir', 
        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"), 
        type=str,
    )

    parser.add_argument(
        '--train-data-dir', 
        type=str,
    )

    parser.add_argument(
        '--eval-data-dir', 
        type=str,
    )

    parser.add_argument(
        '--tft-output-dir', 
        type=str,
    )

    parser.add_argument(
        '--learning-rate', 
        default=0.001, 
        type=float
    )

    parser.add_argument(
        '--batch-size', 
        default=512, 
        type=float
    )

    parser.add_argument(
        '--hidden-units', 
        default="64,32", 
        type=str
    )

    parser.add_argument(
        '--num-epochs', 
        default=10, 
        type=int
    )

    return parser.parse_args()


def main():
    args = get_args()
    
    hyperparams = vars(args)
    hyperparams = defaults.update_hyperparams(hyperparams)
    logging.info(f"Hyperparameter: {hyperparams}")
    logging.info("")
    
    classifier = trainer.train(
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        raw_schema_dir=RAW_SCHEMA_DIR,
        tft_output_dir=args.tft_output_dir,
        hyperparams=hyperparams,
        log_dir=args.log_dir,
    )
    
    exporter.export_serving_model(
        classifier=classifier,
        serving_model_dir=args.model_dir,
        raw_schema_dir=RAW_SCHEMA_DIR,
        tft_output_dir=args.tft_output_dir,
    )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Python Version = {sys.version}')
    logging.info(f'TensorFlow Version = {tf.__version__}')
    logging.info(f'TF_CONFIG = {os.environ.get("TF_CONFIG", "Not found")}')
    logging.info(f'DEVICES = {device_lib.list_local_devices()}')
    main()

