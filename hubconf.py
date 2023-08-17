# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from neuralcompression.zoo import msillm_quality_1 as _msillm_quality_1
from neuralcompression.zoo import msillm_quality_2 as _msillm_quality_2
from neuralcompression.zoo import msillm_quality_3 as _msillm_quality_3
from neuralcompression.zoo import msillm_quality_4 as _msillm_quality_4
from neuralcompression.zoo import msillm_quality_5 as _msillm_quality_5
from neuralcompression.zoo import msillm_quality_6 as _msillm_quality_6
from neuralcompression.zoo import noganms_quality_1 as _noganms_quality_1
from neuralcompression.zoo import noganms_quality_2 as _noganms_quality_2
from neuralcompression.zoo import noganms_quality_3 as _noganms_quality_3
from neuralcompression.zoo import noganms_quality_4 as _noganms_quality_4
from neuralcompression.zoo import noganms_quality_5 as _noganms_quality_5
from neuralcompression.zoo import noganms_quality_6 as _noganms_quality_6

dependencies = ["torch"]


def msillm_quality_1(pretrained=True, **kwargs):
    """
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.035 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_1(pretrained=pretrained, **kwargs)


def msillm_quality_2(pretrained=True, **kwargs):
    """
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.07 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_2(pretrained=pretrained, **kwargs)


def msillm_quality_3(pretrained=True, **kwargs):
    r"""
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.14 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_3(pretrained=pretrained, **kwargs)


def msillm_quality_4(pretrained=True, **kwargs):
    """
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.3 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_4(pretrained=pretrained, **kwargs)


def msillm_quality_5(pretrained=True, **kwargs):
    """
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.45 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_5(pretrained=pretrained, **kwargs)


def msillm_quality_6(pretrained=True, **kwargs):
    """
    Pretrained MS-ILLM model

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.9 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _msillm_quality_6(pretrained=pretrained, **kwargs)


def noganms_quality_1(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.035 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_1(pretrained=pretrained, **kwargs)


def noganms_quality_2(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.07 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_2(pretrained=pretrained, **kwargs)


def noganms_quality_3(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.14 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_3(pretrained=pretrained, **kwargs)


def noganms_quality_4(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.3 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_4(pretrained=pretrained, **kwargs)


def noganms_quality_5(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.45 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_5(pretrained=pretrained, **kwargs)


def noganms_quality_6(pretrained=True, **kwargs):
    """
    Pretrained No-GAN model with HiFiC Mean-Scale Hyperprior architecture.

    This model was trained for the paper:

    MJ Muckley, A El-Nouby, K Ullrich, H Jegou, J Verbeek.
    Improving Statistical Fidelity for Neural Image Compression with Implicit
    Local Likelihood Models. In *ICML*, 2023.

    The target bitrate is 0.9 bits per pixel

    pretrained (bool): kwargs, load pretrained weights into the model
    """

    return _noganms_quality_6(pretrained=pretrained, **kwargs)
