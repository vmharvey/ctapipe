from abc import abstractmethod
import numpy as np
from ..core.component import TelescopeComponent
from ..core.traits import FloatTelescopeParameter, BoolTelescopeParameter

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
    Abstract class for configurable image modifying algorithms. Use
    ``ImageModifier.from_name()`` to construct an instance of a particular algorithm
    """

    @abstractmethod
    def __call__(self, tel_id: int, image: np.ndarray) -> np.ndarray:
        """
        Modifies a given image

        Returns
        -------
        np.ndarray
            modified image
        """
        pass


class ImageSmearer(ImageModifier):
    """
    Smear everything
    """

    smear_factor = FloatTelescopeParameter(
        default_value=0.2, help="Fraction of light to move to each neighbor"
    ).tag(config=True)

    def __call__(self, tel_id, image):
        return smear_image(self.subarray.tel[tel_id], self.smear_factor[tel_id])


class NoiseAdder(ImageModifier):
    # is there a way to turn this off?
    max_threshold = FloatTelescopeParameter(
        default_value=1e10, help="maximum charge in photoelectrons to add noise"
    ).tag(config=True)
    min_threshold = FloatTelescopeParameter(
        default_value=0.0, help="minimum charge in photoelectrons to add noise"
    ).tag(config=True)
    # this only allows the same noise for all pixels
    # ToDo: Think about whether we want per pixel values and if so how to configure it
    noise_level = FloatTelescopeParameter(
        default_value=5.0, help="minimum charge in photoelectrons to add noise"
    ).tag(config=True)
    correct_bias = BoolTelescopeParameter(
        default_value=True, help="If True subtract the expected noise from the image"
    ).tag(config=True)

    def __call__(self, tel_id, image, rng=None):
        mask = (image > self.min_threshold[tel_id]) & (
            image < self.max_threshold[tel_id]
        )
        return add_noise(
            image,
            noise_level=self.noise_level[tel_id] * mask.astype(np.float32),
            rng=rng,
            correct_bias=self.correct_bias,
        )


class LSTNoiseAdder(NoiseAdder):
    """
    Add in everything lstchain does.
    This is probably much slower because its not done in one go.
    Need to benchmark maybe
    """

    transition_charge = FloatTelescopeParameter(
        default_value=8, help="maximum charge in photoelectrons to add noise"
    ).tag(config=True)
    dim_pixel_bias = FloatTelescopeParameter(
        default_value=0.6, help="maximum charge in photoelectrons to add noise"
    ).tag(config=True)
    bright_pixel_noise = FloatTelescopeParameter(
        default_value=1.44, help="maximum charge in photoelectrons to add noise"
    ).tag(config=True)

    def __call__(self, tel_id, image, rng=None):
        dim_pixel_mask = (image > self.min_threshold[tel_id]) & (
            image < self.transition_threshold[tel_id]
        )
        bright_pixel_mask = (image < self.max_threshold[tel_id]) & (
            image > self.transition_threshold[tel_id]
        )
        increased_noise_dim = (
            add_noise(
                image,
                noise_level=self.noise_level[tel_id]
                * dim_pixel_mask.astype(np.float32),
                rng=rng,
                correct_bias=self.correct_bias,
            )
            + self.dim_pixel_bias[tel_id]
        )
        return add_noise(
            increased_noise_dim,
            noise_level=self.bright_pixel_noise[tel_id]
            * bright_pixel_mask.astype(np.float32),
            rng=rng,
            correct_bias=self.correct_bias,
        )


class PixelValueSetter(ImageModifier):
    pass
