from event_aug.noise import gen_perlin_2d


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
