# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import ToTensor
from tqdm import tqdm

from neuralcompression.metrics import (
    MultiscaleStructuralSimilarity,
    calc_psnr,
    pickle_size_of,
    update_patch_fid,
)


def rescale_image(image: Tensor, back_to_float: bool = True) -> Tensor:
    dtype = image.dtype
    image = (image * 255 + 0.5).to(torch.uint8)

    if back_to_float:
        image = image.to(dtype)

    return image


def main():
    parser = ArgumentParser()

    parser.add_argument("clic_path", type=str, help="path to CLIC2020 directory")

    args = parser.parse_args()
    clic_path = Path(args.clic_path)

    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/NeuralCompression", "msillm_quality_3")
    model = model.to(device)
    model = model.eval()
    model.update()
    model.update_tensor_devices("compress")

    totensor = ToTensor()

    msssim_metric = MultiscaleStructuralSimilarity(data_range=255.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    psnr_vals = []
    bpp_vals = []

    for image_path in tqdm(list(clic_path.glob("*.png"))):
        with open(image_path, "rb") as f:
            image_pil = Image.open(f)
            image_pil = image_pil.convert("RGB")

        image = totensor(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            compressed = model.compress(image, force_cpu=False)
            decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

        num_bytes = pickle_size_of(compressed)
        bpp = num_bytes * 8 / (image.shape[0] * image.shape[-2] * image.shape[-1])
        bpp_vals.append(float(bpp))

        orig_image = rescale_image(image)
        pred_image = rescale_image(decompressed)

        with torch.no_grad():
            update_patch_fid(image, decompressed, fid_metric)

            orig_image = rescale_image(image)
            pred_image = rescale_image(decompressed)

            psnr_val = calc_psnr(pred_image, orig_image)
            psnr_vals.append(float(psnr_val))
            msssim_metric(pred_image, orig_image)
            lpips_metric(decompressed, image)

    bpp_total = sum(bpp_vals) / len(bpp_vals)
    psnr_total = sum(psnr_vals) / len(psnr_vals)
    msssim_total = float(msssim_metric.compute())
    lpips_total = float(lpips_metric.compute())
    fid_total = float(fid_metric.compute())

    print("Compression complete")
    print(f"Rate: {bpp_total}")
    print(f"PSNR: {psnr_total}")
    print(f"MS-SSIM: {msssim_total}")
    print(f"LPIPS: {lpips_total}")
    print(f"FID: {fid_total}")


if __name__ == "__main__":
    main()
