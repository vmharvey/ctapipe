from abc import abstractmethod
import numpy as np
from numba import njit
from ..core.component import TelescopeComponent
from ..core.traits import FloatTelescopeParameter, BoolTelescopeParameter
from ..instrument import PixelShape
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


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
        noisy_image -= noise_level
    return noisy_image


@njit(cache=True)
def smear_psf_randomly(
    image, fraction, indices, indptr, smear_probabilities, seed=None
):
    """
    Create a new image with values smeared across the
    neighbor pixels specified by `indices` and `indptr`.
    These are what defines the sparse neighbor matrix
    and are available as attributes from the neighbor matrix.
    The amount of charge that is distributed away from a given
    pixel is drawn from a poissonian distribution.
    The distribution of this charge among the neighboring
    pixels follows a multinomial.
    Pixels at the camera edge lose charge this way.
    No geometry is available in this function due to the numba optimization,
    so the indices, indptr and smear_probabilities have to match
    to get sensible results.

    Parameters:
    -----------
    image: ndarray
        1d array of the pixel charge values
    fraction: float
        fraction of charge that will be distributed among neighbors (modulo poissonian)
    indices: ndarray[int]
        CSR format index array of the neighbor matrix
    indptr: ndarray[int]
        CSR format index pointer array of the neighbor matrix
    smear_probabilities: ndarray[float]
        shape: (n_neighbors, )
        A priori distribution of the charge amongst neighbors.
        In most cases probably of the form np.full(n_neighbors, 1/n_neighbors)
    seed: int
        Random seed for the numpy rng.
        Because this is numba optimized, a rng instance can not be used here

    Returns:
    --------
    new_image: ndarray
        1d array with smeared values
    """
    new_image = image.copy()
    np.random.seed(seed)

    for pixel in range(len(image)):

        if image[pixel] <= 0:
            continue

        to_smear = np.random.poisson(image[pixel] * fraction)
        if to_smear == 0:
            continue

        # remove light from current pixel
        new_image[pixel] -= to_smear

        # add light to neighbor pixels
        neighbors = indices[indptr[pixel] : indptr[pixel + 1]]
        n_neighbors = len(neighbors)

        # all neighbors are equally likely to receive the charge
        # we always distribute the charge into 6 neighbors, so that charge
        # on the edges of the camera is lost
        neighbor_charges = np.random.multinomial(to_smear, smear_probabilities)

        for n in range(n_neighbors):
            neighbor = neighbors[n]
            new_image[neighbor] += neighbor_charges[n]

    return new_image


def smear_psf_statically(image, geom, smear_factor):
    """
    Create a new image with values smeared to the direct pixel neighbors
    Pixels at the camera edge lose charge this way.
    This behaves differently for square and hexagonal pixels:
    - For hexagonal pixels a fraction of light equal to smear_factor
    in each pixel gets shifted to the neighboring pixels. Each pixel receives 1/6
    of the light
    - For square pixels, the image is converted to a 2d-array and a 3x3 gaussian
    kernel is applied. Less light diffuses to diagonal neighbors. Smear factor
    is the standard deviation of the gaussian kernel in this case

    Parameters:
    -----------
    image: ndarray
        1d array of the pixel charge values
    geom: ctapipe.instrument.CameraGeometry
    fraction: float
        fraction of charge that will be distributed among neighbors

    Returns:
    --------
    smeared_image: ndarray
    """
    if geom.pix_type is PixelShape.HEXAGON:
        max_neighbors = 6
        diffused_image = (
            (image * geom.neighbor_matrix).sum(axis=1) * smear_factor / max_neighbors
        )
        remaining_image = image * (1 - smear_factor)
        smeared_image = remaining_image + diffused_image
    elif geom.pix_type is PixelShape.SQUARE:
        image_2d = geom.to_regular_image(image)
        # construct a normalized 3x3 kernel for convolution
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1
        kernel = gaussian_filter(kernel, sigma=smear_factor)

        smeared_2d = convolve2d(image_2d, kernel, mode="same")
        smeared_image = geom.regular_image_to_1d(smeared_2d)
    else:
        raise Exception(f"Unknown pixel type {geom.pix_type}")
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


class NSBNoiseAdder(ImageModifier):
    """
    Component to tune simulated background to
    overserved NSB values.
    A differentiation between bright and dim pixels is taking place
    because this happens at DL1a level and in general the integration window
    would change for peak-searching extraction algorithms with different background levels
    introducing a bias to the charge in dim pixels.

    The performed steps include:
    - Smearing of the image (simulating a worse PSF)
    - Adding poissonian noise (different for bright and dim pixels)
    - Adding a bias to dim pixel charges (see above)
    """

    smear_factor = FloatTelescopeParameter(
        default_value=0.0, help="Fraction of light to move to each neighbor"
    ).tag(config=True)
    transition_charge = FloatTelescopeParameter(
        default_value=0.0, help="separation between dim and bright pixels"
    ).tag(config=True)
    dim_pixel_bias = FloatTelescopeParameter(
        default_value=0.0, help="extra bias to add in dim pixels"
    ).tag(config=True)
    dim_pixel_noise = FloatTelescopeParameter(
        default_value=0.0, help="expected extra noise in dim pixels"
    ).tag(config=True)
    bright_pixel_noise = FloatTelescopeParameter(
        default_value=0.0, help="expected extra noise in bright pixels"
    ).tag(config=True)
    correct_bias = BoolTelescopeParameter(
        default_value=True,
        help="If True subtract the expected noise from the image. possibly unclear what this does with dim_pixel_bias",
    ).tag(config=True)

    def __call__(self, tel_id, image, rng=None):
        smeared_image = smear_psf_statically(
            image,
            self.subarray.tel[tel_id].camera.geometry,
            self.smear_factor.tel[tel_id],
        )
        # TODO: how to get max n_neighbors?
        # geom = self.subarray.tel[tel_id].camera.geometry
        # smeared_image = smear_psf_randomly(
        #    image,
        #    self.smear_factor.tel[tel_id],
        #    geom.neighbor_matrix_sparse.indices,
        #    geom.neighbor_matrix_sparse.indptr,
        #    np.full(
        #        geom.neighbor_matrix.sum(axis=1).max(),
        #        1 / geom.neighbor_matrix.sum(axis=1).max()
        #    )
        # )

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
