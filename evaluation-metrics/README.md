[<u><small><🎯Back to Homepage></small></u>](/README.md)

<div align="center">

# Evaluation Metric Document
This document mainly illustrates how to compute metric scores for evaluation.

</div>

- [<u>1. FID Score</u>](#generate-data-filelist)
- [<u>2. CLIP Score</u>](#extract-coco-captions)

Execute the following command line to compute FID and CLIP scores based on generated results:
```bash
python evaluation-metrics.py --image_path IMAGE_PATH_OF_COCO_2017_VAL --caption_path CAPTION_PATH_OF_COCO_2017_VAL --sample_path GENERATED_RESULTS --outdir OUTPUT_PATH
```
You can refer to this example command line:
```bash
python evaluation-metrics.py --image_path data/coco2017val/images --caption_path data/coco2017val/coco-captions --sample_path outputs/generated-results --outdir outputs/evaluation-metrics
```

<!-- omit in toc -->
# FID Score
We follow [this implementation](https://github.com/mseitzer/pytorch-fid) to compute FID score. 

<!-- omit in toc -->
# CLIP Score
We follow [this issue](https://github.com/TencentARC/T2I-Adapter/issues/62) in [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) to compute the CLIP score, where the original implementation is shown as follows:
```python
image_features = model.encode_image(image)
text_features = model.encode_text(text)
image_features = image_features / image_features.norm(dim=1, keepdim=True)
text_features = text_features / text_features.norm(dim=1, keepdim=True)
sim = F.cosine_similarity(image_features, text_features)
sim = sim.cpu().data.numpy().astype(np.float32)
```
For the ViT version, we use `ViT-L/14`.