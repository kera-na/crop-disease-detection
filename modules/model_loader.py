import os
import json
import tensorflow as tf


def load_deployment(folder):

    model_path = os.path.join(
        folder,
        "crop_model.keras"
    )

    class_path = os.path.join(
        folder,
        "class_names.json"
    )

    config_path = os.path.join(
        folder,
        "model_config.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path
    )

    with open(class_path) as f:
        classes = json.load(f)

    config = {}

    if os.path.exists(config_path):

        with open(config_path) as f:
            config = json.load(f)

    return model, classes, config


def discover_models(root="deployment_model"):

    models = {}

    if not os.path.exists(root):
        return models

    for folder in os.listdir(root):

        path = os.path.join(root, folder)

        if os.path.isdir(path):

            model_file = os.path.join(
                path,
                "crop_model.keras"
            )

            if os.path.exists(model_file):
                models[folder] = path

    return models
