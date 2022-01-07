from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

MS_SSIM_FACTORS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(
    window_size: int,
    std: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Constructs a 2D Gaussian window, as used in the MS-SSIM paper."""

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"Window size must be positive odd number, not {window_size}")

    indices = torch.arange(
        -(window_size // 2), window_size // 2 + 1, device=device, dtype=dtype
    )

    # Ignoring the constant factor in the Gaussian formula, since
    # the probabilities are immediately normalized below.
    gaussian_probabilities = torch.exp(-0.5 * (indices / std).pow(2))

    normalized = gaussian_probabilities / gaussian_probabilities.sum()
    window = normalized[:, None] * normalized[None, :]
    return window.view(1, 1, window_size, window_size)


def _get_reduction_op(reduction: str) -> Callable[[Tensor], Tensor]:
    if reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        return lambda x: x
    else:
        raise ValueError(f"Unknown reduction type: '{reduction}'")


def _ssim_single_channel(
    x: Tensor,
    y: Tensor,
    data_range: float,
    window_size: int,
    k1: float,
    k2: float,
    gaussian_std: float,
) -> Tensor:
    """
    Computes the SSIM and contrast-structure product on a single channel image.

    At the lowest resolution, the entire SSIM score is used in MS-SSIM
    calculations, while at all other resolutions only the constrast-structure
    product is used.

    Args:
        x: a batch of single-channel images with shape [batch_size, 1,
            height, width].
        y: a batch of single-channel images with shape [batch_size, 1,
            height, width].
        data_range: dynamic range of the input tensors.
        window_size: window size for SSIM calculation.
        k1: k1 parameter for SSIM calculation.
        k2: k2 parameter for SSIM calculation.
        gaussian_std: standard deviation of Gaussian filter to use in SSIM
            calculations.

    Returns:
        Tensor of shape [batch_size, 2], where output[:,0] is SSIM scores
        and output[:,1] is constrast-structure product scores.
    """

    window = _gaussian_window(window_size, gaussian_std, device=x.device, dtype=x.dtype)

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ux = F.conv2d(x, window)
    uy = F.conv2d(y, window)

    uxx = F.conv2d(x * x, window)
    uyy = F.conv2d(y * y, window)
    uxy = F.conv2d(x * y, window)

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    # Equation (6) in MS-SSIM paper.
    a1, a2, b1, b2 = (
        2 * ux * uy + c1,
        2 * vxy + c2,
        ux ** 2 + uy ** 2 + c1,
        vx + vy + c2,
    )

    luminance = a1 / b1
    contrast_structure = a2 / b2

    batch_size = luminance.size(0)
    ssim = (luminance * contrast_structure).view(batch_size, -1).mean(dim=1)
    cs = contrast_structure.view(batch_size, -1).mean(dim=1)

    return torch.stack([ssim, cs])


def _ssim_multichannel(
    x: Tensor,
    y: Tensor,
    data_range: float,
    window_size: int,
    k1: float,
    k2: float,
    gaussian_std: float,
) -> Tensor:
    """
    Applies SSIM independently to each channel.

    Given a multi-channel image, applies single-channel SSIM calculations
    to each channel independently, then stacks the per-channel results
    along a new axis.

    Args:
        x: a batch of images of shape [batch_size, num_channels,
            height, width].
        y: a batch of images of shape [batch_size, num_channels,
            height, width].
        data_range: dynamic range of the input tensors.
        window_size: window size for SSIM calculation.
        k1: k1 parameter for SSIM calculation.
        k2: k2 parameter for SSIM calculation.
        gaussian_std: standard deviation of Gaussian filter to use in SSIM
            calculations.

    Returns:
        Tensor of shape [num_channels, batch_size, 2], where output[i] is
        the output of _ssim_single_channel applied to channel i.
    """

    num_channels = x.shape[1]
    return torch.stack(
        [
            _ssim_single_channel(
                x=x[:, i : i + 1],
                y=y[:, i : i + 1],
                window_size=window_size,
                data_range=data_range,
                k1=k1,
                k2=k2,
                gaussian_std=gaussian_std,
            )
            for i in range(num_channels)
        ]
    )


def _pad_and_downsample(x: Tensor) -> Tensor:
    """
    Downsamples using average pooling, padding if necessary.

    To match the Tensorflow MS-SSIM implementation, we need to pad
    images before pooling if their height/width does not evenly
    divide 2. Tensorflow uses "SYMMETRIC" padding, which is
    equivalent to PyTorch's replication padding in our special
    case where at most 1 element of padding is added to each
    dimension.

    Args:
        x: a batch of images to pad and downsample, of shape [batch_size,
            num_channels, height, width].

    Returns:
        The padded and downsampled tensor, of shape [batch_size,
        num_channels, ceil(height / 2), ceil(width / 2)] where the
        output height/widht is relative to the input shape.
    """

    padded = F.pad(x, (0, x.shape[3] % 2, 0, x.shape[2] % 2), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=2)


def multiscale_structural_similarity(
    preds: Tensor,
    target: Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    gaussian_std: float = 1.5,
    power_factors: Sequence[float] = MS_SSIM_FACTORS,
    reduction: str = "mean",
) -> Tensor:
    """
    Computes the multi-scale structural similarity index measure.

    Follows the algorithm in the paper: Wang, Zhou, Eero P. Simoncelli,
    and Alan C. Bovik. "Multiscale structural similarity for image
    quality assessment." Signals, Systems and Computers, 2004.
    https://www.cns.nyu.edu/pub/eero/wang03b.pdf

    Args:
        preds: a batch of target image approximations.
        target: a batch of images to compare preds against.
        data_range: dynamic range of the input tensors.
        window_size: window size for SSIM calculation.
        k1: k1 parameter for SSIM calculation.
        k2: k2 parameter for SSIM calculation.
        gaussian_std: standard deviation of Gaussian filter to use in SSIM
            calculations.
        power_factors: relative importance of each scale; defaults to
            the values proposed in the paper; the length of
            power_factors determines how many scales to consider.
        reduction: specifies the reduction to apply across batches:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the sum of the output will be
            divided by the number of elements in the output, ``'sum'``: the
            output will be summed.

    Returns:
        The MS-SSIM score between ``preds`` and ``target``, reduced across
        batches as specified by the ``reduction`` argument.
    """

    num_scales = len(power_factors)

    scale_outputs = []
    for scale in range(num_scales):
        if scale != 0:
            preds = _pad_and_downsample(preds)
            target = _pad_and_downsample(target)

        if preds.shape[2] < window_size or preds.shape[3] < window_size:
            raise RuntimeError(
                f"After {scale} rounds of downsampling, the "
                f"images to compare have a shape of {preds.shape}, which is "
                f"smaller than the window size of {window_size}. Please "
                "use larger images or a smaller window size."
            )

        # relu improves numerical stability, cs cannot be negative
        # but might be very close to zero
        scale_outputs.append(
            F.relu(
                _ssim_multichannel(
                    x=preds,
                    y=target,
                    data_range=data_range,
                    window_size=window_size,
                    k1=k1,
                    k2=k2,
                    gaussian_std=gaussian_std,
                )
            ).pow(power_factors[scale])
        )

    # shape is [num_scales, num_channels, 2 (ssim, cs), batch size]
    weighted_outputs = torch.stack(scale_outputs)

    # shape is [num_channels, batch_size]
    ms_ssim = weighted_outputs[-1, :, 0] * torch.prod(
        weighted_outputs[:-1, :, 1], dim=0
    )

    # mean over channels, reduction across batches
    reduction_op = _get_reduction_op(reduction)
    return reduction_op(ms_ssim.mean(dim=0))
