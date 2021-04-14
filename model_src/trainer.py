"""Train and evaluate the model."""

import os
import logging
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow import keras

try:
    from model_src import data, model
except:
    import data, model 


def train(
    train_data_dir,
    eval_data_dir,
    raw_schema_dir,
    tft_output_dir,
    hyperparams,
    log_dir,
):
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    summary_writer.set_as_default()
    summary_writer.init()
    
    logging.info(f"Loading raw schema from {raw_schema_dir}")
    raw_schema = tfdv.load_schema_text(raw_schema_dir)
    raw_feature_spec = schema_utils.schema_as_feature_spec(raw_schema).feature_spec

    logging.info(f"Loading tft output from {tft_output_dir}")
    tft_output = tft.TFTransformOutput(tft_output_dir)
    transformed_feature_spec = tft_output.transformed_feature_spec()
    
    train_dataset = data.get_dataset(
        train_data_dir,
        transformed_feature_spec,
        hyperparams["batch_size"],
    )
    
    eval_dataset = data.get_dataset(
        eval_data_dir,
        transformed_feature_spec,
        hyperparams["batch_size"],
    )
    
    optimizer = keras.optimizers.Adam(
        learning_rate=hyperparams["learning_rate"])
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    
    classifier = model.create_binary_classifier(
        tft_output, hyperparams)
    
    classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logging.info("Model training started...")
    classifier.fit(
        train_dataset,
        epochs=hyperparams["num_epochs"],
        validation_data=eval_dataset,
        callbacks=[early_stopping]
    )
    logging.info("Model training completed.")
    
    return classifier