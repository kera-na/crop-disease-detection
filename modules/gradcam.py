"""Grad-CAM helpers that work with custom CNNs and transfer-learning models
(for example a frozen MobileNetV2 base inside a Sequential or Functional model)."""

import numpy as np
import tensorflow as tf


def _output_rank(layer):
    """Number of output dimensions of a layer, or None if it cannot be determined."""
    try:
        shape = layer.output.shape
    except Exception:
        shape = getattr(layer, "output_shape", None)
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    try:
        return len(shape)
    except Exception:
        return None


def find_last_conv_layer(model):
    """Return the name of the last layer that outputs a 4D feature map.

    Convolution layers and nested feature extractors (such as a MobileNetV2 base)
    are preferred. Any other 4D layer is used only as a fallback.
    """
    layers = list(model.layers)

    for layer in reversed(layers):
        if _output_rank(layer) != 4:
            continue
        class_name = layer.__class__.__name__.lower()
        is_conv = "conv" in class_name
        is_nested_model = len(getattr(layer, "layers", []) or []) > 0
        if is_conv or is_nested_model:
            return layer.name

    for layer in reversed(layers):
        if _output_rank(layer) == 4 and "input" not in layer.__class__.__name__.lower():
            return layer.name

    return None


def _prepare_input(img_array):
    x = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if len(x.shape) == 3:
        x = tf.expand_dims(x, axis=0)
    return x


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _conv_and_grads_graph(x, model, layer_name):
    """Preferred method: a model that returns both the conv features and predictions."""
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])

    with tf.GradientTape() as tape:
        # Watching the input is essential. When the base model is frozen
        # (trainable=False) nothing else is tracked and the gradient is None.
        tape.watch(x)
        conv_out, preds = grad_model(x, training=False)
        conv_out = _first(conv_out)
        preds = _first(preds)
        class_idx = tf.argmax(preds[0])
        score = tf.gather(preds, class_idx, axis=1)

    return conv_out, tape.gradient(score, conv_out)


def _conv_and_grads_sequential(x, model, layer_name):
    """Fallback: run the layers one by one (Sequential or linear models)."""
    conv_out = None
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = x
        for layer in model.layers:
            if layer.__class__.__name__ == "InputLayer":
                continue
            out = layer(out, training=False)
            if layer.name == layer_name:
                conv_out = _first(out)
        preds = _first(out)
        class_idx = tf.argmax(preds[0])
        score = tf.gather(preds, class_idx, axis=1)

    if conv_out is None:
        raise ValueError(f"Layer '{layer_name}' was not found in the model.")

    # Safety check: layer-by-layer execution must reproduce the real model output,
    # otherwise the model is not a simple stack of layers (for example skip connections).
    reference = _first(model(x, training=False))
    if not np.allclose(preds.numpy(), reference.numpy(), atol=1e-4):
        raise ValueError("Model is not a simple linear stack of layers.")
    return conv_out, tape.gradient(score, conv_out)


def make_gradcam_heatmap(img_array, model, layer_name):
    """Return a 2D heatmap (values 0 to 1) for the model's top predicted class."""
    x = _prepare_input(img_array)
    errors = []
    conv_out = grads = None

    for method in (_conv_and_grads_graph, _conv_and_grads_sequential):
        try:
            conv_out, grads = method(x, model, layer_name)
            if grads is not None:
                break
            errors.append(f"{method.__name__}: gradients were None")
        except Exception as exc:
            errors.append(f"{method.__name__}: {exc}")
            conv_out = grads = None

    if grads is None or conv_out is None:
        raise RuntimeError(
            "Grad-CAM could not be generated for this model. "
            + " | ".join(errors)
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy().astype("float32")