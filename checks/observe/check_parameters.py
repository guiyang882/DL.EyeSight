# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt

from eagle.parameter import (
    Binomial, Choice, DiscreteUniform, Normal, Uniform, Deterministic, Clip,
    Discretize, Multiply, Add, Divide, Power, Absolute
)


def main():
    params = [
        ("Binomial(0.1)", Binomial(0.1)),
        ("Choice", Choice([0, 1, 2])),
        ("Choice with p", Choice([0, 1, 2], p=[0.1, 0.2, 0.7])),
        ("DiscreteUniform(0, 10)", DiscreteUniform(0, 10)),
        ("Normal(0, 1)", Normal(0, 1)),
        ("Normal(1, 1)", Normal(1, 1)),
        ("Normal(1, 2)", Normal(0, 2)),
        ("Normal(Choice([-1, 1]), 2)", Normal(Choice([-1, 1]), 2)),
        ("Discretize(Normal(0, 1.0))", Discretize(Normal(0, 1.0))),
        ("Uniform(0, 10)", Uniform(0, 10)),
        ("Deterministic(1)", Deterministic(1)),
        ("Clip(Normal(0, 1), 0, None)", Clip(Normal(0, 1), minval=0, maxval=None)),
        ("Multiply(Uniform(0, 10), 2)", Multiply(Uniform(0, 10), 2)),
        ("Add(Uniform(0, 10), 5)", Add(Uniform(0, 10), 5)),
        ("Absolute(Normal(0, 1))", Absolute(Normal(0, 1)))
    ]

    params_arithmetic = [
        ("Normal(0, 1.0)", Normal(0.0, 1.0)),
        ("Normal(0, 1.0) + 5", Normal(0.0, 1.0) + 5),
        ("5 + Normal(0, 1.0)", 5 + Normal(0.0, 1.0)),
        ("5 + Normal(0, 1.0)", Add(5, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) * 10", Normal(0.0, 1.0) * 10),
        ("10 * Normal(0, 1.0)", 10 * Normal(0.0, 1.0)),
        ("10 * Normal(0, 1.0)", Multiply(10, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) / 10", Normal(0.0, 1.0) / 10),
        ("10 / Normal(0, 1.0)", 10 / Normal(0.0, 1.0)),
        ("10 / Normal(0, 1.0)", Divide(10, Normal(0.0, 1.0), elementwise=True)),
        ("Normal(0, 1.0) ** 2", Normal(0.0, 1.0) ** 2),
        ("2 ** Normal(0, 1.0)", 2 ** Normal(0.0, 1.0)),
        ("2 ** Normal(0, 1.0)", Power(2, Normal(0.0, 1.0), elementwise=True))
    ]

    params_noise = [
        # ("SimplexNoise", SimplexNoise()),
        # ("Sigmoid(SimplexNoise)", Sigmoid(SimplexNoise())),
        # ("SimplexNoise(linear)", SimplexNoise(upscale_method="linear")),
        # ("SimplexNoise(nearest)", SimplexNoise(upscale_method="nearest")),
        # ("FrequencyNoise((-4, 4))", FrequencyNoise(exponent=(-4, 4))),
        # ("FrequencyNoise(-2)", FrequencyNoise(exponent=-2)),
        # ("FrequencyNoise(2)", FrequencyNoise(exponent=2))
    ]

    images_params = [param.draw_distribution_graph() for (title, param) in params]
    images_arithmetic = [param.draw_distribution_graph() for (title, param) in params_arithmetic]

    show_multi_array(images_params)
    show_multi_array(images_arithmetic)


def show_multi_array(image_arrays):
    n = len(image_arrays)
    h, w, c = image_arrays[0].shape
    print("arrays num: {},single image shape: {}".format(n, image_arrays[0].shape))

    if n == 1:
        plt.imshow(image_arrays[0])
        plt.show()
        return

    if int(np.sqrt(n)) ** 2 < n:
        n = int(np.sqrt(n)) + 1
    else:
        n = int(np.sqrt(n))

    large_image = np.zeros((h*n, w*n, c), dtype=image_arrays[0].dtype)
    for i, img in enumerate(image_arrays):
        x1, y1 = (i%n)*w, (i//n)*h
        x2, y2 = (i%n+1)*w, (i//n+1)*h
        large_image[y1:y2, x1:x2] = img
    plt.imshow(large_image)
    plt.show()


if __name__ == "__main__":
    main()
