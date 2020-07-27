import cv2
import argparse
from glob import glob
from os.path import join, isdir
from os import mkdir
from shutil import rmtree
from tqdm import tqdm
import ntpath
import sys
import numpy as np

from dataset.face_aligner import FaceAligner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str,
                        default="/home/igor/datasets/align_celeba/img_align_celeba/img_align_celeba",
                        help="Input data folder")
    parser.add_argument("--output_folder", type=str,
                        default="/home/igor/datasets/faces",
                        help="Output folder")
    parser.add_argument("--target_image_size", type=int, default=64,
                        help="Size (both w and h) of output aligned images [pix]")
    parser.add_argument("--eye_level", type=float, default=0.4,
                        help="Hight of eye level (from top), normalized to frame's height")
    parser.add_argument("--eye_dist", type=float, default=0.3,
                        help="Distance b"
                             "etween eye, normalized to frame's width")
    args = parser.parse_args()

    if isdir(args.output_folder):
        rmtree(args.output_folder)
    mkdir(args.output_folder)

    img_paths = glob(join(args.input_folder, "*.jpg"))
    total_imgs = len(img_paths)
    img_idx = 0
    cnt = 0
    r_sample = []
    g_sample = []
    b_sample = []
    with tqdm(total=total_imgs, desc=f'Image {img_idx + 1}/{total_imgs}', unit='image') as pbar:
        aligner = FaceAligner(target_size=args.target_image_size, eye_level=args.eye_level, eye_dist=args.eye_dist)
        for img_idx, img_path in enumerate(img_paths):
            try:
                img = cv2.imread(img_path)

                aligned_img = aligner(img)
                if aligned_img is not None:
                    cv2.imwrite(join(args.output_folder, ntpath.basename(img_path)), aligned_img)
                    cnt += 1
                    norm_img = np.divide(aligned_img, 255.0)
                    r_sample += norm_img[:, :, 2].flatten().tolist()
                    g_sample += norm_img[:, :, 1].flatten().tolist()
                    b_sample += norm_img[:, :, 0].flatten().tolist()
            except:
                print("Exception has been thrown, img idx:", img_idx, "path:", img_path)
                print("Exception:", sys.exc_info()[0])
            pbar.update(1)
    print("Total images saved:", cnt, "skipped:", total_imgs - cnt)
    r_mean, r_std = np.mean(r_sample), np.std(r_sample)
    g_mean, g_std = np.mean(g_sample), np.std(g_sample)
    b_mean, b_std = np.mean(b_sample), np.std(b_sample)
    text = ""
    with open("dataset_stat.txt", "w") as f:
        text += "Mean (r/g/b): " + str(r_mean) + " " + str(g_mean) + " " + str(b_mean) + "\n"
        text += "Std (r/g/b): " + str(r_std) + " " + str(g_std) + " " + str(b_std) + "\n"
        text += "Total images: " + str(cnt) + "\n"
        text += "Src dataset path: " + args.input_folder + "\n"
        text += "Preprocessed dataset path: " + args.output_folder + "\n"
        text += "Output image size: " + str(args.target_image_size) + "\n"
        f.write(text)
    print(text)
    return


if __name__ == "__main__":
    main()
