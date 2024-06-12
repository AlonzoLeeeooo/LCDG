[<u><small><🎯Back to Homepage></small></u>](/README.md)

<div align="center">

# Configuration Document
This document mainly illustrates how to modify configuration files for various functionalities.

</div>

- [<u>1. Configure Pre-trained Model Weights</u>](#configure-pre-trained-model-weights)
- [<u>2. Configure Training Data</u>](#configure-training-data)

We present an example configuration file [`example.yaml`](example.yaml) for reference.

<!-- omit in toc -->
# Configure Pre-trained Model Weights
To configure pre-trained model weights, you need to modify the following lines `(line 10, 41, and 65)`:
```
model:
  base_learning_rate: 2.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    ckpt_path: SD_CHECKPOINT_PATH                       <----- Replace with SD weights
    first_stage_key: image # "jpg"
    cond_stage_key: caption # "txt"
    image_size: 64
    channels: 4
    cond_stage_trainable: true
    conditioning_key: crossattn
    monitor: val/loss_simple_ema
    scale_factor: 0.18215
    use_ema: False

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 32
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [ 4, 2, 1 ]
        num_res_blocks: 2
        channel_mult: [ 1, 2, 4, 4 ]
        num_heads: 8
        use_spatial_transformer: True
        transformer_depth: 1
        context_dim: 768
        use_checkpoint: True
        legacy: False

    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        ckpt_path: SD_VAE_CHECKPOINT_PATH                 <----- Replace with VAE weights
        embed_dim: 4
        monitor: val/rec_loss
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult:
          - 1
          - 2
          - 4
          - 4
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
      params:
        version: CLIP_CHECKPOINT_PATH                     <----- Replace with CLIP weights
```

<!-- omit in toc -->
# Configure Training Data
To configure the training data, you need to modify the following lines `(line 71, 72, and 73)`:

```
condition_aligner_config:
  mode: from_text
  cond_type: edge
  kmeans_center: 16
  image_dir: COCO_IMAGE_FILELIST                      <----- Replace with image file list
  cond_dir: COCO_CONDITIOIN_PATH                      <----- Replace with condition folder
  text_dir: COCO_CAPTION_PATH                         <----- Replace with caption folder
  val_image_dir: 
  val_cond_dir: 
  val_text_dir: 
  val_diffusion_steps: [500, 1000]
  val_sample_freq: 100
  size: 64
  image_size: 512
  epochs: 100
  blocks: [[2, 4, 8], [2, 4, 8, 12]]
  downsample_factor: 8
  in_channels: 7040
  out_channels: 4
  time_channels: 256
  learning_rate: 0.0001
```
To obtain `COCO_IMAGE_FILELIST` and `COCO_CAPTION_PATH`, please refer to [this document](/tools/README.md) with our automatic scripts.