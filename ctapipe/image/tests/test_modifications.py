import numpy as np
from numpy.testing import assert_allclose
from ctapipe.instrument import CameraGeometry
from ctapipe.image import modifications

# perform tests with methods as well as components!


def test_mask_broken():
    geom = CameraGeometry.from_name("LSTCam")
    broken_pixel_ids = [1, 5, 10]

    expected = np.ones_like(geom.pix_id, dtype=bool)
    expected[broken_pixel_ids] = 0
    mask = modifications.mask_broken(geom, broken_pixel_ids)

    assert (mask == expected).all()


def test_mask_thresholds():
    image = np.array([1, 2, 3, 4, 5])
    mask = modifications.mask_thresholds(image, 4.5, 2.1)
    expected = [False, False, True, True, False]

    assert (mask == expected).all()


def test_add_noise():
    image = np.array([0, 0, 5, 1, 0, 0])
    rng = np.random.default_rng(42)
    # test different noise per pixel:
    noise = [6, 8, 0, 7, 9, 12]
    noisy = modifications.add_noise(image, noise, rng, correct_bias=False)
    assert image[2] == noisy[2]
    # For other seeds there exists a probability > 0 for no noise added at all
    assert noisy.sum() > image.sum()

    # test scalar
    noisy = modifications.add_noise(image, 20, rng, correct_bias=False)
    diff_no_bias = noisy - image
    assert (noisy > image).all()

    # test bias
    noisy = modifications.add_noise(image, 20, rng, correct_bias=True)
    assert np.sum(diff_no_bias) > np.sum(noisy - image)


def test_smear_image():
    geom = CameraGeometry.from_name("LSTCam")
    image = np.zeros_like(geom.pix_id, dtype=np.float64)
    # select two pixels, one at the edge with only 5 neighbors
    # Thats why we divide by 6 below
    signal_pixels = [1, 1853]
    neighbors = geom.neighbor_matrix[signal_pixels]
    for signal_value in [1, 5]:
        image[signal_pixels] = signal_value
        for s in [0, 0.2, 1]:
            smeared_light = s * signal_value
            smeared = modifications.smear_image(image, geom, s)
            assert np.isclose((image.sum() - smeared.sum() - smeared_light / 6), 0)
            neighbors_1 = smeared[neighbors[0]]
            neighbors_1853 = smeared[neighbors[1]]
            assert_allclose(neighbors_1, smeared_light / 6)
            assert_allclose(neighbors_1853, smeared_light / 6)
            assert_allclose(smeared[signal_pixels], signal_value - smeared_light)
