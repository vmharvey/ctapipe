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


class NullModifier(ImageModifier):
    def __call__(self, tel_id: int, image: np.ndarray) -> np.ndarray:
        return image


class LSTImageModifier(ImageModifier):
    """
    Add in everything lstchain does.
    """

    smear_factor = FloatTelescopeParameter(
        default_value=0.2, help="Fraction of light to move to each neighbor"
    ).tag(config=True)
    transition_charge = FloatTelescopeParameter(
        default_value=8, help="separation between dim and bright pixels"
    ).tag(config=True)
    dim_pixel_bias = FloatTelescopeParameter(
        default_value=0.6, help="extra bias to add in dim pixels"
    ).tag(config=True)
    dim_pixel_noise = FloatTelescopeParameter(
        default_value=1.5, help="expected extra noise in dim pixels"
    ).tag(config=True)
    bright_pixel_noise = FloatTelescopeParameter(
        default_value=1.44, help="expected extra noise in bright pixels"
    ).tag(config=True)
    correct_bias = BoolTelescopeParameter(
        default_value=True,
        help="If True subtract the expected noise from the image. possibly unclear what this does with dim_pixel_bias",
    ).tag(config=True)

    def __call__(self, tel_id, image, rng=None):
        smeared_image = smear_image(
            image,
            self.subarray.tel[tel_id].camera.geometry,
            self.smear_factor.tel[tel_id],
        )
        noise = np.where(
            image > self.transition_charge.tel[tel_id],
            self.bright_pixel_noise.tel[tel_id],
            self.dim_pixel_noise.tel[tel_id],
        )
        image_with_noise = add_noise(
            smeared_image, noise, rng=rng, correct_bias=self.correct_bias.tel[tel_id]
        )
        bias = np.where(
            image > self.transition_charge.tel[tel_id],
            0,
            self.dim_pixel_noise.tel[tel_id],
        )
        return (image_with_noise + bias).astype(np.float32)
