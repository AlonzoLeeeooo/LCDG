<p align="center">
  <h1 align="center">Late-Constraint Diffusion Guidance for Controllable Image Synthesis</h1>
The official code implementation of "Late-Constraint Diffusion Guidance for Controllable Image Synthesis".

[[Paper]]([github_materials/tissor.jpg](https://arxiv.org/abs/2305.11520)) / [[Project]](https://alonzoleeeooo.github.io/LCDG/) / [Model Weights] / [Demo]

# News
- We have uploaded the training and testing code of LCDG. Afterwards, we would also release our pre-trained model weights as well as an interactive demo. **Star the project to get notified!**

# Overview
![tissor](github_materials/tissor.jpg)
Diffusion models, either with or without text condition, have demonstrated impressive capability in synthesizing photorealistic images given a few or even no words. These models may not fully satisfy user need, as normal users or artists intend to control the synthesized images with specific guidance, like overall layout, color, structure, object shape, and so on. To adapt diffusion models for controllable image synthesis, several methods have been proposed to incorporate the required conditions as regularization upon the intermediate features of the diffusion denoising network. These methods, known as early-constraint ones in this paper, have difficulties in handling multiple conditions with a single solution. They intend to train separate models for each specific condition, which require much training cost and result in non-generalizable solutions. To address these difficulties, we propose a new approach namely late-constraint: we leave the diffusion networks unchanged, but constrain its output to be aligned with the required conditions. Specifically, we train a lightweight condition adapter to establish the correlation between external conditions and internal representations of diffusion models. During the iterative denoising process, the conditional guidance is sent into corresponding condition adapter to manipulate the sampling process with the established correlation. We further equip the introduced late-constraint strategy with a timestep resampling method and an early stopping technique, which boost the quality of synthesized image meanwhile complying with the guidance. Our method outperforms the existing early-constraint methods and generalizes better to unseen condition.

# To-Do Lists
- [ ] Online demo of LCDG.
- [ ] Pre-trained model weights.
- [x] Official instructions of installation and usage of LCDG.
- [x] Training code of LCDG.
- [x] Testing code of LCDG.

# Prerequisites
We integrate the basic environment to run both of the training and testing code in `environment.sh` using `pip` as package manager. Simply running `bash environment.sh` would get the required packages installed.

# Before Training or Testing
## 1. Prepare Pre-trained Model Weights of Stable Diffusion
Before running the code, pre-trained model weights of the diffusion models and corresponding VQ models should be prepared locally. For `v1.4` or `CelebA` model weights, you could refer to the GitHub repository of [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and excute their provided scripts to download by running:
```bash
bash scripts/download_first_stage.sh
bash scripts/download_models.sh
```
## 2. Modify the Configuration Files
We provide example configuration files of edge, color and mask conditions in `configs/lcdg`. Modify `line 5 and line 43` in these configuration files with the corresponding paths of pre-trained model weights.

# Training the Condition Adapter
## 1. Prepare Your Training Data
As is reported in our [paper](https://arxiv.org/abs/2305.11520), our condition adapter could be well-trained with 10,000 randomly collected images. You could formulate your data directory in the following structure:
```bash
collected_dataset/
├── bdcn_edges
├── captions
└── images
```
For `edge` condition, we use [bdcn](https://github.com/pkuCactus/BDCN) to generate the supervision signals from source images. For `mask` condition, we use [u2net](https://github.com/xuebinqin/U-2-Net) to detect saliency masks as supervision. For `color` condition, we use simulated color stroke as supervision and have incorparated corresponding code in `condition_adaptor_src/condition_adaptor_dataset.py`.

## 2. Modify the Configuration Files
After the training data is ready, you need to modify `line 74 to 76` with the corresponding paths of the training data. Additionally, if evaluation is required, you need to modify `line 77 to 79` with corresponding paths of the splitted validation data.

## 3. Starting Training
Now you are ready to go by executing `condition_adaptor_train.py`, such as:
```bash
python condition_adaptor_train.py
    -b /path/to/config/file
    -l /path/to/output/path
```

# Inferencing with Trained Condition Adapter
After hours of training, you could try sampling images with the trained condition adapter. We provide an example execution command as follows:
```bash
python sample_single_image.py
    --base /path/to/config/path
    --indir /path/to/target/condition
    --caption "text prompt"
    --outdir /path/to/output/path
    --resume /path/to/condition/adapter/weights
```

# Qualitative Comparison
We demonstrate the qualitative comparisons upon different conditions in the following figures.
<details><summary>Canny Edge</summary>
<div align="center">
<img src="github_materials/canny_edge.jpg">
</div>
</details>

<details><summary>HED Edge</summary>
<div align="center">
<img src="github_materials/hed_edge.jpg">
</div>
</details>

<details><summary>User Sketch</summary>
<div align="center">
<img src="github_materials/user_sketch.jpg">
</div>
</details>

<details><summary>Color Storke</summary>
<div align="center">
<img src="github_materials/color_stroke.jpg">
</div>
</details>

<details><summary>Image Palette</summary>
<div align="center">
<img src="github_materials/image_palette.jpg">
</div>
</details>

<details><summary>Mask</summary>
<div align="center">
<img src="github_materials/mask.jpg">
</div>
</details>

# License
This work is licensed under MIT license. See the [LICENSE](LICENSE) for details.

# Citation
If you find our work enlightening or the codebase is helpful to your work, please cite our paper:
```bibtex
@misc{liu2023lateconstraint,
    title={Late-Constraint Diffusion Guidance for Controllable Image Synthesis}, 
    author={Chang Liu and Dong Liu},
    year={2023},
    eprint={2305.11520},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
# Acknowledgements
This codebase is heavily built upon the source code of [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Thanks to their great implementations!

# Stars and Forked
[![Stargazers repo roster for @AlonzoLeeeooo/LCDG](https://reporoster.com/stars/dark/AlonzoLeeeooo/LCDG)](https://github.com/AlonzoLeeeooo/LCDG/stargazers)

[![Forkers repo roster for @AlonzoLeeeooo/LCDG](https://reporoster.com/forks/dark/AlonzoLeeeooo/LCDG)](https://github.com/AlonzoLeeeooo/LCDG/network/members)


<p align="center">
    <a href="https://api.star-history.com/svg?repos=AlonzoLeeeooo/LCDG&type=Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=AlonzoLeeeooo/LCDG&type=Date" alt="Star History Chart">
    </a>
<p>