# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import torch


def create_input(shape, offset: int = 0):
    x = np.arange(np.prod(shape)).reshape(shape) + offset

    return torch.from_numpy(x).to(torch.get_default_dtype())


def create_one_image(shape, seed):
    rng = np.random.default_rng(seed)
    image = create_input(shape)

    # this test won't work for completely random warps - about a 10% error
    # so we create a single vector flow and rotate it randomly
    flow = np.stack(
        (
            np.ones(shape[0] * shape[-2] * shape[-1]),
            np.zeros(shape[0] * shape[-2] * shape[-1]),
        )
    ) * (rng.uniform() / 10)

    angle = rng.uniform() * 2 * np.pi
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rot_mat = np.array(((cos_val, -sin_val), (sin_val, cos_val)))
    flow = np.reshape(np.transpose(rot_mat @ flow), (shape[0], shape[-2], shape[-1], 2))
    flow = torch.tensor(flow).to(image)

    tf_image = tf.convert_to_tensor(image.permute(0, 2, 3, 1).numpy())
    tf_flow = tf.convert_to_tensor(
        np.flip(flow.numpy(), -1) * ((np.array(shape[-2:]) - 1) / 2)
    )
    tf_image_warp = tfa.image.dense_image_warp(tf_image, -tf_flow)

    tf_image_warp = torch.tensor(tf_image_warp.numpy()).permute(0, 3, 1, 2)

    return {"image": image, "flow": flow, "tf_image_warp": tf_image_warp}


def main():
    shapes = [(3, 3, 72, 64), (5, 3, 55, 18), (6, 3, 73, 35)]
    seeds = [0, 1, 2]
    images = []
    for shape, seed in zip(shapes, seeds):
        images.append(create_one_image(shape, seed))

    with open("dense_image_warp.pkl", "wb") as f:
        pickle.dump(images, f)


if __name__ == "__main__":
    main()
