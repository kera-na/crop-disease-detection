import numpy as np


def preprocess_image(
        image,
        width,
        height):

    img = image.resize(
        (
            width,
            height
        )
    )

    array = np.array(
        img,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


def predict_image(
        image,
        model,
        class_names):

    width = model.input_shape[2]
    height = model.input_shape[1]

    array = preprocess_image(
        image,
        width,
        height
    )

    predictions = model.predict(
        array,
        verbose=0
    )[0]

    index = np.argmax(predictions)

    return {

        "class":
            class_names[index],

        "confidence":
            float(
                predictions[index]
            ),

        "probabilities":
            predictions,

        "top3":
            np.argsort(
                predictions
            )[::-1][:3]
    }