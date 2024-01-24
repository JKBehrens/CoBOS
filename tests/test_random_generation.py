import numpy as np


def test_randon_number_gen():
    gen = np.random.default_rng(1)

    res = gen.random(10)

    gen = np.random.default_rng(1)
    res2 = gen.random(10)

    assert np.allclose(res, res2)

    res3 = gen.random(10)

    assert not np.allclose(res2, res3)