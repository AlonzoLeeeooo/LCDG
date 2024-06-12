[<u><small><🎯Back to Homepage></small></u>](/README.md)

<div align="center">

# Toolkit Document
This document mainly illustrates how to use toolkit scripts to prepare the basic essentials of the training data.

</div>

- [<u>1. Generate Data Filelist</u>](#generate-data-filelist)
- [<u>2. Extracted COCO Captions</u>](#extract-coco-captions)


<!-- omit in toc -->
# Generate Data Filelist
Execute the following command line to generate the data filelist according to the images in a folder:
```bash
python tools/generate-data-filelist.py --indir IMAGE_PATH --outdir DATA_FILELIST_PATH
```
The generated data filelist (in `.txt` format) will be saved in `DATA_FILELIST_PATH/data_flist.txt`.
You can refer to this example command line:
```bash
python tools/generate-data-filelist.py --indir data/coco2017val/images --outdir data/coco2017val/data_flist
```

<!-- omit in toc -->
# Extract COCO Captions
1. Download the official annotation file of COCO captions from [this link](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) and unzip it.
2. Execute the following command line to extract COCO captions from the official annotation files:
```bash
python tools/extract-coco-captions.py --annotation-file COCO_OFFICIAL_ANNOTATION_FILE --outdir OUTPUT_PATH
```
The extracted captions will be saved in `OUTPUT_PATH`, where the filenames are correspond to `image_id` in the official annotation file.
You can refer to this example command line:
```bash
python tools/extract-coco-captions.py --annotation-file COCO_OFFICIAL_ANNOTATION_FILE --outdir OUTPUT_PATH
```