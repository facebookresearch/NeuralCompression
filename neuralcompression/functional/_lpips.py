# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lpips as lpips_package
from torch import Tensor

from ._multiscale_structural_similarity import _get_reduction_op


def _load_lpips_model(
    base_network: str, linear_weights_version: str, use_linear_calibration: bool
):
    model = lpips_package.LPIPS(
        net=base_network,
        version=linear_weights_version,
        lpips=use_linear_calibration,
        verbose=False,
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model


def lpips(
    preds: Tensor,
    target: Tensor,
    base_network: str = "alex",
    use_linear_calibration: bool = True,
    linear_weights_version: str = "0.1",
    normalize: bool = False,
    reduction: str = "mean",
) -> Tensor:
    """
    Learned perceptual image patch similarity.

    Implementation of the LPIPS metric introduced in "The Unreasonable
    Effectiveness of Deep Features as a Perceptual Metric" by Richard Zhang,
    Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.

    Note:
        The input images to this function as assumed to have values in the
        range [-1,1], NOT in the range [0,1]. If image values are in the range
        [0,1], pass ``normalize=True``.

    Args:
        preds: Predicted images with shape (batch_size, channel, height, width)
        target: Expected images with shape (batch_size, channel, height, width)
        base_network: the pretrained architecture to extract features from.
            Must be one of ``'alex'``, ``'vgg'``, ``'squeeze'``,
            corresponding to AlexNet, VGG, and SqueezeNet, respectively.
        use_linear_calibration: whether to use pretrained weights to
            compute a weighed average across model layers. If ``False``,
            different layers are averaged.
        linear_weights_version: which pretrained linear weights version
            to use. Must be one of ``'0.0'``, ``'0.1'``. This is ignored if
            ``use_linear_calibration==False``.
        normalize: whether to rescale the input images from the
            range [0,1] to the range [-1,1] before computing LPIPS.
        reduction: specifies the reduction to apply across batches:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.

    Returns:
        The LPIPS score between ``preds`` and ``target``, reduced across
        batches as specified by the ``reduction`` argument.
    """

    if base_network not in ("alex", "vgg", "squeeze"):
        raise ValueError(
            f"Unknown base network {base_network}"
            " - please pass one of 'alex', 'vgg', or 'squeeze'."
        )

    if linear_weights_version not in ("0.0", "0.1"):
        raise ValueError(
            f"Unknown linear weights version {linear_weights_version}"
            " - please pass '0.0' or '0.1'."
        )

    model = _load_lpips_model(
        base_network=base_network,
        linear_weights_version=linear_weights_version,
        use_linear_calibration=use_linear_calibration,
    )
    outputs = model(preds, target, normalize=normalize).view(-1)
    reduction_op = _get_reduction_op(reduction)
    return reduction_op(outputs)
