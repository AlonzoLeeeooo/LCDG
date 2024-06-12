import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--annotation-file', default='', type=str)
parser.add_argument('--outdir', default='', type=str)

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

with open(args.annotation_file, 'r') as j:
    annotation_file = json.load(j)

images = annotation_file["images"]
captions = annotation_file["annotations"]

count = 0

print(f'\nThere are {len(captions)} captions to generate...\n')

for item in captions:
    count += 1
    print(f"Progress: {count}/{len(captions)}")
    
    image_id = item["image_id"]
    caption = item["caption"]
    filename = str(image_id).rjust(12, '0') + '.txt'
    
    txt = open(os.path.join(args.outdir, filename), "w")
    txt.write(caption)
    txt.close()