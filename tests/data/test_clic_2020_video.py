# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import pytorchvideo.data.clip_sampling
from utils import create_random_image

from neuralcompression.data import CLIC2020Video


@pytest.fixture
def data(tmp_path):
    num_frames = 9
    videos = [
        "Animation_1080P-01b3",
        "Animation_1080P-05f8",
        "Animation_1080P-0c4f",
    ]

    for video in videos:
        directory = tmp_path.joinpath("CLIC2020Video").joinpath(video)

        directory.mkdir(parents=True)

        for index in range(num_frames):
            path = directory.joinpath(f"{index}_y.png")

            create_random_image(path, (3, 224, 224))

    clip_sampler = pytorchvideo.data.clip_sampling.make_clip_sampler("random", 4)

    data = CLIC2020Video(
        tmp_path.joinpath("CLIC2020Video"),
        clip_sampler=clip_sampler,
    )

    return data, len(videos)


class TestCLIC2020Video:
    def test___getitem__(self, data):
        data, _ = data

        assert isinstance(data[0], dict)

    def test___len__(self, data):
        data, n = data

        assert len(data) == n
