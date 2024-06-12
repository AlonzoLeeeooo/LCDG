[<u><small><🎯Back to Homepage></small></u>](/README.md)

<div align="center">

# Data Pre-Processing Document
This document mainly illustrates how to use several off-the-shelf toolkits to automatically perform data pre-processing.

</div>

- [<u>1. Overview</u>](#overview)
- [<u>2. Prepare Datasets</u>](#prepare-datasets)
- [<u>3. Generate Canny Edge</u>](#generate-canny-edge)
- [<u>4. Generate HED Edge</u>](#generate-hed-edge)
- [<u>5. Generate Color Stroke</u>](#generate-color-stroke)
- [<u>6. Generate Image Palette</u>](#generate-image-palette)
- [<u>7. Generate Saliency Mask</u>](#generate-saliency-mask)

<!-- omit in toc -->
# Overview

We use the following toolkits to perform data pre-processing of LaCon:
|Condition|Script|Model Weights|
|---|---|---|
|Canny Edge|`generate-canny.py`|-|
|HED Edge|`bdcn-edge-detection/generate-bdcn-edge.py`|`toolkits/bdcn.pth`|
|Color Stroke|`generate-stroke.pth`|-|
|Image Palette|`generate-palette.py`|-|
|Saliency Mask|`u2net-saliency-detection/generate-saliency-mask.py`|`toolkits/u2net.pth`|

Before you start data pre-processing with the above toolkits, please download the toolkit model weights from our [Huggingface repo](https://huggingface.co/AlonzoLeeeooo/LaCon) and place them in `bdcn-edge-detection/checkpoints` and `u2net-saliency-detection/checkpoints`.

<!-- omit in toc -->
# Prepare Datasets
Before you start to prepare the conditions, you need to download the source images of different datasets. You can refer to the following links:
- [COCO 2014 training set](http://images.cocodataset.org/zips/train2014.zip)
- [COCO 2017 validation set](http://images.cocodataset.org/zips/val2017.zip)
- [CelebA-HQ](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)

<!-- omit in toc -->
# Generate Canny Edge
Execute the following command line to extract Canny edge maps from images in a folder:
```bash
python data-preprocessing/generate-canny.py --indir IMAGE_PATH --outdir CANNY_PATH --threshold1 CANNY_THRESHOLD_ONE --threshold2 CANNY_THRESHOLD_TWO
```
The extracted Canny edge maps will be saved in `CANNY_PATH`.
You can refer to this example command line:
```bash
python data-preprocessing/generate-canny.py --indir data/coco2017val/images --outdir data/coco2017val/canny-edges --threshold1 200 --threshold2 225
```

<!-- omit in toc -->
# Generate HED Edge
1. Download the model weights of [BDCN edge extractor](https://github.com/pkuCactus/BDCN) from [this link](https://huggingface.co/AlonzoLeeeooo/LaCon/tree/main), and place the weights in `data-preprocessing/bdcn-edge-detection/checkpoints`.
2. Execute the following command line to extract HED edge maps from images in a folder:
```bash
python data-preprocessing/bdcn-edge-detection/generate-bdcn-edge.py --indir IMAGE_PATH --outdir HED_EDGE_PATH
```
The extracted HED edge maps will be saved in `HED_EDGE_PATH`.
You can refer to this example command line:
```bash
python data-preprocessing/bdcn-edge-detection/generate-bdcn-edge.py --indir data/coco2017val/images --outdir data/coco2017val/bdcn-edges
```

<!-- omit in toc -->
# Generate Color Stroke
Execute the following command line to extract color strokes from images in a folder:
```bash
python data-preprocessing/generate-stroke.py --indir IMAGE_PATH --outdir OUTPUT_PATH --kmeans_center K_MEANS_CENTER_NUMBER
```
You can refer to this example command line:
```bash
python data-preprocessing/generate-stroke.py --indir data/coco2017val/images --outdir data/coco2017val/color-strokes --kmeans_center 16
```
By changing `--kmeans_center`, you can control the number of color in the generated strokes; by turning on `--visualize_intermediate`, you can visualize the intermediate results (i.e., source image, filtered image, and generated strokes) during color stroke generation. The intermediate results will be saved in `OUTPUT_PATH/intermediates`.

<!-- omit in toc -->
# Generate Image Palette
Execute the following command line to extract image palettes from images in a folder:
```bash
python data-preprocessing/generate-palette.py --indir IMAGE_PATH --outdir OUTPUT_PATH --size BICUBIC_SIZE --image_resolution IMAGE_RESOLUTION
```
You can refer to this example command line:
```bash
python data-preprocessing/generate-palette.py --indir data/coco2017val/images --outdir data/coco2017val/image-palette --size 8 --image_resolution 512
```
By changing `--size`, you can control the image size of Bicubic down-sampled result; by changing `--image_resolution`, you can define the original image resolution; by turning on `--visualize_intermediate`, you can visualize the intermediate results (i.e., source image, Bicubic down-sampled image, and generated palette) during image palette generation. The intermediate results will be saved in `OUTPUT_PATH/intermediates`.

<!-- omit in toc -->
# Generate Saliency Mask
1. Download the model weights of [U$^2$-Net](https://github.com/xuebinqin/U-2-Net) from [this link](https://huggingface.co/AlonzoLeeeooo/LaCon/tree/main), and place the weights in `u2net-saliency-detection/checkpoints`.
2. Execute the following command line to extract saliency masks from images in a folder:
```bash
python data-preprocessing/u2net-saliency-detection/generate-saliency-mask.py --indir IMAGE_PATH --outdir OUTPUT_PATH
```
You can refer to this example command line:
```bash
python data-preprocessing/u2net-saliency-detection/generate-saliency-mask.py --indir data/coco2017val/images --outdir data/coco2017val/saliency-masks
```