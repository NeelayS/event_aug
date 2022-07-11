from event_aug.noise import gen_fractal_3d, gen_perlin_2d, gen_perlin_3d


def test_gen_perlin_2d():

    noise = gen_perlin_2d(
        shape=(32, 32),
        scale=100,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=None,
    )
    assert noise.shape == (32, 32)

    noise = gen_perlin_2d(
        shape=(32, 32),
        scale=100,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=None,
        reshape_size=(16, 16),
        crop_size=(8, 8),
    )
    assert noise.shape == (8, 8)


def test_gen_fractal_3d():

    noise = gen_fractal_3d((2, 8, 8), res=(1, 4, 4))
    assert noise.shape == (2, 8, 8)

    noise = gen_fractal_3d(
        (2, 8, 8), res=(1, 4, 4), reshape_size=(4, 4), crop_size=(2, 2)
    )
    assert noise.shape == (2, 2, 2)


def test_gen_perlin_3d():

    noise = gen_perlin_3d((2, 8, 8), res=(1, 4, 4))
    assert noise.shape == (2, 8, 8)

    noise = gen_perlin_3d((2, 8, 8), res=(1, 4, 4), reshape_size=(4, 4), crop_size=(2, 2))
    assert noise.shape == (2, 2, 2)
