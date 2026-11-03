"""Model loading, preprocessing and batched crops for the marker positivity CNN."""

import numpy as np
from scipy import ndimage

#: Counts above the brightest neighbour that make a pixel hot, as in training.
HOT_PIXEL_THRESHOLD = 50.0


def load_positivity_model(tf, path):
    """Load the checkpoint with the custom layer it was trained with."""

    class GroupNormalization(tf.keras.layers.Layer):
        """Normalize each image independently in groups of channels (NHWC)."""

        def __init__(self, groups=8, epsilon=1e-5, **kwargs):
            super().__init__(**kwargs)
            self.groups = groups
            self.epsilon = epsilon

        def build(self, input_shape):
            channels = input_shape[-1]
            if channels is None or channels % self.groups:
                raise ValueError("Input channels must be known and divisible by groups")
            self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones")
            self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros")
            super().build(input_shape)

        def call(self, inputs):
            shape = tf.shape(inputs)
            grouped = tf.reshape(inputs, [shape[0], shape[1], shape[2], self.groups, -1])
            mean, variance = tf.nn.moments(grouped, axes=[1, 2, 4], keepdims=True)
            normalized = (grouped - mean) * tf.math.rsqrt(variance + self.epsilon)
            normalized = tf.reshape(normalized, shape)
            return normalized * self.gamma + self.beta

        def get_config(self):
            return {**super().get_config(), "groups": self.groups, "epsilon": self.epsilon}

    return tf.keras.models.load_model(
        path, compile=False,
        custom_objects={"GroupNormalization": GroupNormalization,
                        "CellMarker>GroupNormalization": GroupNormalization})


def remove_hot_pixels(plane, threshold=HOT_PIXEL_THRESHOLD):
    """Apply Bodenmiller's neighbour filter once to one (H, W) marker channel.

    Pixels more than ``threshold`` raw counts above all eight spatial neighbours
    are replaced with their brightest neighbour, with mirrored image borders.
    The result is float32; only neighbouring values are ever written, so it casts
    back to the image's integer type unchanged.
    """
    values = np.asarray(plane, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Marker data contain NaN or infinite values")
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    neighbour_max = ndimage.maximum_filter(values, footprint=footprint, mode="mirror")
    return np.where(values - neighbour_max > threshold, neighbour_max, values)


def make_positivity_batch(markers, cells, cell_ids, origins, z, crop,
                          n_side, use_other_cells):
    """Gather many cells at once, preserving training channel order and counts.

    ``origins`` contains the already-clamped top-left (y, x) of each crop.
    Only the neighbouring marker channels are read, so temporary memory does
    not grow with the number of channels in the original image.

    Images are (Y, X) here, but the model was trained and is run in the notebooks
    on (X, Y) arrays, so each crop is laid out with X as its first axis.
    """
    n_markers = 2 * n_side + 1
    batch = np.zeros((len(cell_ids), crop, crop,
                      n_markers + 1 + int(use_other_cells)), dtype=np.float32)
    offsets = np.arange(crop)
    ys = origins[:, 0, None, None] + offsets[None, None, :]
    xs = origins[:, 1, None, None] + offsets[None, :, None]

    first = max(0, z - n_side)
    last = min(markers.shape[2], z + n_side + 1)
    dest = first - (z - n_side)
    batch[..., dest:dest + last - first] = markers[ys, xs, first:last]

    patch = cells[ys, xs]
    target = patch == cell_ids[:, None, None]
    batch[..., n_markers] = target
    if use_other_cells:
        batch[..., n_markers + 1] = (patch > 0) & ~target
    return batch
