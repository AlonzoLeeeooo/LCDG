# LaCon for Diffusers

基于Diffusers框架实现的LaCon (Late-Constraint Diffusion)可控图像生成模型。这是对原始[AlonzoLeeeooo/LCDG](https://github.com/AlonzoLeeeooo/LCDG)仓库的现代化重新实现。

## 📖 简介

LaCon (Late-Constraint Diffusion) 是一种用于可控图像生成的新颖方法，它通过在扩散过程的后期阶段施加约束来实现精确的条件控制。与传统的早期约束方法不同，LaCon使用条件对齐器(Condition Aligner)来将扩散模型的中间特征与目标条件进行对齐。

### 主要特性

- 🎨 **多种条件类型支持**: 边缘、遮罩、颜色描边、图像调色板等
- 🚀 **高效训练**: 基于预训练的Stable Diffusion模型微调
- 🔧 **Diffusers兼容**: 完全兼容Hugging Face Diffusers生态系统
- 📊 **灵活控制**: 可调节的条件强度和截断步数
- 🎯 **渐进约束**: 在采样过程中逐步减少条件约束

## 🚀 快速开始

### 安装依赖

```bash
# 创建虚拟环境
conda create -n lacon python=3.8
conda activate lacon

# 安装PyTorch (根据您的CUDA版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy tqdm tensorboard
```

### 基本使用

```python
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler

# 导入我们的自定义组件
from diffusers_lacon import LaConPipeline, ConditionAligner

# 加载预训练的Stable Diffusion模型
base_pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
)

# 初始化条件对齐器
condition_aligner = ConditionAligner(
    time_channels=256,
    in_channels=2560,  # 根据特征块调整
    out_channels=4,    # VAE潜在通道数
)

# 创建LaCon管道
pipeline = LaConPipeline(
    vae=base_pipeline.vae,
    text_encoder=base_pipeline.text_encoder,
    tokenizer=base_pipeline.tokenizer,
    unet=base_pipeline.unet,
    scheduler=DDIMScheduler.from_config(base_pipeline.scheduler.config),
    condition_aligner=condition_aligner,
)

# 移动到GPU
pipeline = pipeline.to("cuda")

# 创建条件图像(例如边缘图)
condition_image = Image.open("edge_map.png")

# 生成图像
result = pipeline(
    prompt="a beautiful landscape with mountains and a lake",
    condition_image=condition_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    condition_scale=2.0,
    height=512,
    width=512,
    truncation_steps=25,  # 前25步应用条件
)

# 保存结果
result.images[0].save("generated_image.png")
```

## 🎯 支持的条件类型

### 1. 边缘控制
- **Canny边缘**: 精确的边缘检测结果
- **HED边缘**: 更平滑的边缘表示
- **用户草图**: 手绘线条

### 2. 遮罩控制
- **显著性遮罩**: 基于目标区域的生成
- **用户涂鸦**: 自由形式的遮罩

### 3. 颜色控制
- **颜色描边**: 指定区域的颜色约束
- **图像调色板**: 基于参考颜色的生成

## 🛠️ 训练自定义条件对齐器

### 数据准备

创建以下目录结构：

```
data/
├── images/           # 原始图像
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── conditions/       # 条件图像
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── captions/        # 文本描述 (可选)
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### 训练命令

```bash
python -m diffusers_lacon.training.train_condition_aligner \
    --image_dir ./data/images \
    --condition_dir ./data/conditions \
    --caption_dir ./data/captions \
    --condition_type edge \
    --output_dir ./outputs \
    --logging_dir ./logs \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --image_size 512
```

### 训练参数说明

- `--condition_type`: 条件类型 (edge, mask, color, stroke)
- `--batch_size`: 批次大小，根据GPU内存调整
- `--learning_rate`: 学习率
- `--truncation_steps`: 训练时的条件截断步数
- `--condition_scale`: 条件约束强度

## 📊 推荐设置

根据不同条件类型的推荐设置：

| 条件类型 | 条件强度 | 截断步数 | 推荐用途 |
|---------|---------|---------|---------|
| Canny边缘 | 2.0 | 25 | 精确的结构控制 |
| HED边缘 | 2.5 | 25 | 平滑的边缘引导 |
| 用户草图 | 2.0 | 30 | 创意绘画辅助 |
| 显著性遮罩 | 2.0 | 30 | 区域生成控制 |
| 颜色描边 | 2.0 | 30 | 颜色布局指导 |
| 图像调色板 | 2.0 | 40 | 整体色调控制 |

## 🔧 高级使用

### 自定义特征提取

```python
from diffusers_lacon.utils.feature_extractor import UNetFeatureExtractor

# 使用真实的UNet特征提取
feature_extractor = UNetFeatureExtractor(
    unet=pipeline.unet,
    feature_blocks=[[2, 4, 8], [2, 4, 8, 12]]
)

# 在管道中使用
pipeline.feature_extractor = feature_extractor
```

### 批量生成

```python
# 批量生成不同条件的图像
conditions = [edge_image, mask_image, color_image]
prompts = ["landscape", "portrait", "abstract art"]

for i, (condition, prompt) in enumerate(zip(conditions, prompts)):
    result = pipeline(
        prompt=prompt,
        condition_image=condition,
        condition_scale=2.0,
        truncation_steps=25,
    )
    result.images[0].save(f"output_{i}.png")
```

### 条件强度调节

```python
# 不同强度的条件控制
for scale in [1.0, 2.0, 3.0]:
    result = pipeline(
        prompt="a beautiful garden",
        condition_image=edge_image,
        condition_scale=scale,
        truncation_steps=25,
    )
    result.images[0].save(f"scale_{scale}.png")
```

## 📈 性能优化

### GPU内存优化

```python
# 启用注意力切片以减少内存使用
pipeline.enable_attention_slicing()

# 启用顺序CPU卸载
pipeline.enable_sequential_cpu_offload()

# 使用半精度
pipeline = pipeline.to(torch.float16)
```

### 推理加速

```python
# 使用DPM-Solver调度器加速
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)

# 减少推理步数
result = pipeline(
    prompt="landscape",
    condition_image=condition,
    num_inference_steps=20,  # 减少步数
    condition_scale=2.0,
)
```

## 🎨 示例画廊

### 边缘控制生成

| 条件 | 生成结果 | 提示词 |
|------|----------|--------|
| ![边缘图](examples/edge_condition.png) | ![生成图](examples/edge_result.png) | "a mountain landscape at sunset" |

### 遮罩控制生成

| 条件 | 生成结果 | 提示词 |
|------|----------|--------|
| ![遮罩图](examples/mask_condition.png) | ![生成图](examples/mask_result.png) | "a cat sitting in a garden" |

## 🔬 技术细节

### 架构概述

1. **条件对齐器**: 多层卷积网络，用于将UNet特征映射到条件空间
2. **特征提取**: 从UNet的中间层提取多尺度特征
3. **梯度引导**: 通过反向传播计算条件对齐梯度
4. **渐进约束**: 在采样过程中逐步减少条件强度

### 与原始实现的区别

- ✅ 使用Diffusers框架，更易于集成
- ✅ 模块化设计，支持自定义组件
- ✅ 改进的特征提取机制
- ✅ 优化的训练流程
- ✅ 更好的内存效率

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 减少批次大小
   batch_size = 1
   
   # 启用内存优化
   pipeline.enable_attention_slicing()
   pipeline.enable_sequential_cpu_offload()
   ```

2. **条件效果不明显**
   ```python
   # 增加条件强度
   condition_scale = 3.0
   
   # 增加截断步数
   truncation_steps = 35
   ```

3. **生成质量差**
   ```python
   # 使用更多推理步数
   num_inference_steps = 50
   
   # 调整引导强度
   guidance_scale = 7.5
   ```

## 📚 参考文献

```bibtex
@misc{liu-etal-2024-lacon,
      title={{LaCon: Late-Constraint Diffusion for Steerable Guided Image Synthesis}}, 
      author={{Chang Liu, Rui Li, Kaidong Zhang, Xin Luo, and Dong Liu}},
      year={2024},
      eprint={2305.11520},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目基于原始LaCon论文的方法实现，遵循相应的开源许可证。

## 🙏 致谢

- 原始LaCon论文作者: Chang Liu, Rui Li, Kaidong Zhang, Xin Luo, Dong Liu
- Hugging Face团队的Diffusers库
- Stable Diffusion社区
