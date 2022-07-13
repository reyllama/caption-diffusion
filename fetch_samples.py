import shutil
import os
import random
from os import path as PATH
import argparse

def main(args):

    img_files = os.listdir(args.path)
    samples = random.sample(img_files, args.n_sample)
    os.makedirs(args.target, exist_ok=True)
    cnt = 0
    for sample in samples:
        try:
            path_src = PATH.join(args.path, sample)
            path_tar = PATH.join(args.target, sample)
            shutil.copy(path_src, path_tar)
            cnt += 1
        except:
            pass
    print(f"Copied {cnt} files to {args.target}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", type=str, help="image directory path")
    parser.add_argument("-t", "--target", type=str, help="save path")
    parser.add_argument("-n", "--n_sample", type=int, help="how many images to sample")

    args = parser.parse_args()
    main(args)