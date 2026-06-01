import logging

import numpy as np

# import pyclesperanto as cle
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.filters import threshold_local as sk_threshold_local
from skimage.filters import threshold_otsu as sk_threshold_otsu
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.segmentation import expand_labels, watershed

# from .utils import preprocess_image  # TODO: restore when utils.py exists

logger = logging.getLogger(__name__)


def _gauss_otsu_labeling(
    image, outline_sigma: float = 2, threshold: float = 1, **kwargs
):
    """Gauss-Otsu-Labeling can be used to segment objects such
    as nuclei with bright intensity on low intensity background images.

    The outline_sigma parameter allows tuning how precise segmented
    objects are outlined. Under the hood, this filter applies a
    Gaussian blur, Otsu-thresholding and connected component labeling.

    See also
    --------
    .. [0] https://github.com/clEsperanto/pyclesperanto_prototype/
    blob/master/demo/segmentation/gauss_otsu_labeling.ipynb
    """
    image = np.asarray(image)

    # blur
    blurred_outline = gaussian(image, outline_sigma)
    if "debug" in kwargs:
        np.save("blurred_outline.npy", blurred_outline)

    # threshold
    sk_threshold = sk_threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold * sk_threshold

    # connected component labeling
    labels = label(binary_otsu)

    return labels


def _voronoi_otsu_labeling(
    image,
    spot_sigma: float = 2,
    outline_sigma: float = 2,
    mode: str = "local",
    threshold_size: int = 101,
    **kwargs,
):
    """Voronoi-Otsu-Labeling is a segmentation algorithm for
    blob-like structures such as nuclei and granules with high
    signal intensity on low-intensity background.

    The two sigma parameters allow tuning the segmentation result.
    The first sigma controls how close detected cells can be
    (spot_sigma) and the second controls how precise segmented
    objects are outlined (outline_sigma). Under the hood,
    this filter applies two Gaussian blurs, spot detection,
    Otsu-thresholding and Voronoi-labeling. The thresholded
    binary image is flooded using the Voronoi approach starting
    from the found local maxima. Noise-removal sigma for spot
    detection and thresholding can be configured separately.

    This allows segmenting connected objects such as not to dense nuclei.
    If the nuclei are too dense, consider using stardist [1] or cellpose [2].

    See also
    --------
    .. [0] https://github.com/clEsperanto/pyclesperanto_prototype
    /blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb
    .. [1] https://www.napari-hub.org/plugins/stardist-napari
    .. [2] https://www.napari-hub.org/plugins/cellpose-napari
    """
    image = np.asarray(image)

    # blur and detect local maxima
    blurred_spots = gaussian(image, spot_sigma)
    spot_centroids = local_maxima(blurred_spots)

    # blur and threshold
    blurred_outline = gaussian(image, outline_sigma)
    if "debug" in kwargs:
        np.save("blurred_outline.npy", blurred_outline)

    if mode == "local":
        blurred_outline = (
            (blurred_outline - blurred_outline.min())
            / (blurred_outline.max() - blurred_outline.min())
            * 255
        )
        sk_threshold = sk_threshold_local(blurred_outline, threshold_size, offset=0)
        binary_otsu = blurred_outline > sk_threshold + 1
    elif mode == "global":
        sk_threshold = sk_threshold_otsu(blurred_outline)
        binary_otsu = blurred_outline > sk_threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_otsu

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_otsu, labeled_spots, mask=binary_otsu)

    return labels


# @preprocess_image  # TODO: restore when utils.py exists
def run(
    image,
    labeller="voronoi",
    spot_sigma=2,
    outline_sigma=2,
    expand=2,
    threshold=1,
    **kwargs,
):
    """Run the watershed segmentation algorithm on an image.
    Parameters
    ----------
    image : array
        Image array.
    labeller : str, optional
        Labeller to use, "voronoi" or "gauss", by default "voronoi".
    spot_sigma : float, optional
        Sigma for spot detection, by default 2.
    outline_sigma : float, optional
        Sigma for outline detection, by default 2.
    expand : int, optional
        Expand labels by this distance, by default 2.
    threshold : float, optional
        Threshold for the labeller, by default 1.

    Returns
    -------
    array
        Segmented image.
    """

    nuclei_gauss = ndi.gaussian_filter(image, sigma=1.5)
    nuclei_gauss = ndi.convolve(
        nuclei_gauss, np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    )

    logger.info(f"Running labeller {labeller}")
    if labeller == "voronoi":
        labels = _voronoi_otsu_labeling(
            nuclei_gauss,
            spot_sigma=spot_sigma,
            outline_sigma=outline_sigma,
            threshold=threshold,
            **kwargs,
        )
    elif labeller == "gauss":
        labels = _gauss_otsu_labeling(
            nuclei_gauss, outline_sigma=outline_sigma, threshold=threshold, **kwargs
        )
    else:
        logger.error(f"Invalid labeller {labeller}")

    if expand > 0:
        labels = expand_labels(labels, distance=expand)

    return labels
