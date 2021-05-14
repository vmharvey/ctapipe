from abc import abstractmethod
import numpy as np
from ..core.component import TelescopeComponent

# the most important part is the selection api!!


def mask_broken(geom, pix_ids):
    """
    Create a boolean mask with deselected pixels
    Does this really need a function?
    """
    mask = np.ones(shape=geom.pix_id.shape, dtype=bool)
    mask[pix_ids] = 0
    return mask


def mask_thresholds(image, upper=None, lower=None):
    """
    Create a boolean mask with pixels selected based on thresholds.
    Does this really need a function?
    """
    mask = np.ones_like(image, dtype=bool)
    if upper:
        mask[image > upper] = 0
    if lower:
        mask[image < lower] = 0
    return mask


def add_noise(image, noise_level, rng=None, correct_bias=True):
    """
    Create a new image with added poissonian noise
    """
    # notes:
    # mask and thresholds is probably not very useful
    # mask is only useful if noise level is a scalar, could also provide the array directly
    # maybe only noise level and bias, selection in tool?
    if not rng:
        rng = np.random.default_rng()
    noisy_image = image.copy()
    noise = rng.poisson(noise_level)
    noisy_image += noise
    if correct_bias:
        # which one to use?
        # Fluctuations that have no bias per pixel
        noisy_image -= noise_level
        # No bias over the whole image
        # noisy_image -= np.mean(noise)
    return noisy_image


def smear_image(image, geom, smear_factor):
    """
    Create a new image with values smeared to the direct pixel neighbors
    Pixels at the camera edge lose charge this way.
    """
    # make clear what smear factor is supposed to mean and that only direct neighbors are taken into account
    # smear factor can be an array as well -> selection in tool possible!
    # a more sophisticated approach might make use of the pixel area, but thats complicated
    if geom.pix_type.value == "hexagon":
        max_neighbors = 6
    elif geom.pix_type.value == "rectangular":  # thats probably labeld differently
        max_neighbors = 8  # or 4? lookup neighbor matrix for a rect cam
    else:
        print(geom.pix_type)
        raise Exception("Unknown pixel type")

    diffused_image = (
        (image * geom.neighbor_matrix).sum(axis=1) * smear_factor / max_neighbors
    )
    remaining_image = image * (1 - smear_factor)
    smeared_image = remaining_image + diffused_image
    return smeared_image


class ImageModifier(TelescopeComponent):
    """
    ...
    """

    @abstractmethod
    def __call__(self, tel_id: int, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Modifies a given image

        Returns
        -------
        np.ndarray
            modified image
        """
        pass


class ImageSmearer(ImageModifier):
    pass


class NoiseAdder(ImageModifier):
    pass


class PixelValueSetter(ImageModifier):
    pass
