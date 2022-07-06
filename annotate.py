import os
from os import path
import json
import argparse

# Assume that we are on 'img' (fashiongen)

def annotate(args):
    cur_dir = args.path
    out_dir = args.out_path
    out_format = args.out_format # default: json (coco-style annotation)
    annotations = []

    folders = os.listdir(cur_dir)
    num_categories = len(folders)
    for i, folder in enumerate(folders):
        if not os.path.isdir(path.join(cur_dir, folder)):
            continue
        imgs = os.listdir(path.join(cur_dir, folder))
        for img in imgs:
            data = dict()
            data['caption'] = folder.replace("_", " ")
            data['image'] = path.join(folder, img)
            data['image_id'] = data['image'].split('.')[-2].replace("/", "_")
            annotations.append(data)
        if (i+1) % 100 == 0:
            print(f"{i+1} / {num_categories} categories finished")
    if out_format == 'json':
        with open(path.join(out_dir, "fashiongen_captions.json"), "w") as f:
            json.dump(annotations, f)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create annotations")
    parser.add_argument("path", type=str)
    parser.add_argument("--out_path", type=str, default='.')
    parser.add_argument("--out_format", type=str, default='json')
    args = parser.parse_args()
    annotate(args)