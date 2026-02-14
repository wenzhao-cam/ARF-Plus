# Copyright 2021 Alex Yu
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:       sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import sys
sys.path.append("..")

import torch
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import get_expon_lr_func
from util import config_util
from util.util import compute_ssim, viridis_cmap

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import cv2

from icecream import ic
from torchvision import models
import torchvision.transforms as T

# from style_transfer_losses import StyleTransferLosses
from nnfm_loss_spatial_control_semantic_depth import NNFMLoss, match_colors_for_image_set

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

fcn = models.segmentation.fcn_resnet101(pretrained=True).eval().to(device=device)
net = fcn

# parser = argparse.ArgumentParser()
# config_util.define_common_args(parser)
print('-------begin-------')
print('generate semantic mask')


# args.style: ../data/styles/26.jpg
# style_img_path = '../data/styles/26.jpg'

# style_img = imageio.imread(style_img_path).astype(np.float32) / 255.0

# print(style_img)
# item_type = "all"
# item_type = "TV"
# semantic_type = "all"
item_type = "TV"
# targeted_stylization_folder =f"room_7_origin_to_compare_spatial_semantic_{item_type}"
targeted_stylization_folder =f"room_7_spatial_control_semantic_segment_{item_type}"
# semantic_type = "table"
# item_type = "chair"
# if item_type == 'all':
#     select_c_list = [9, 11, 20] # chair + table + tv/monitor
# if item_type == 'TV':
#     select_c_list = [20]
# if item_type == 'table':
#     select_c_list = [11]
# if item_type == 'chair':
#     select_c_list = [9]


scene_type = "llff"
scene_name = "room"
# Horse_122_origin_to_compare_spatial_semantic
# /home/wl301/projects/ARF_CAM/ARF/opt/ckpt_arf/tnt/Horse_122_spatial_control_semantic_segment



semantic_type = "room"
stylized_folder_path = f"ckpt_arf/{scene_type}/{targeted_stylization_folder}/test_renders_path"
photo_folder_path = f"ckpt_svox2/{scene_type}/{scene_name}/test_renders_path"


result_folder = f"ckpt_arf/{scene_type}/{targeted_stylization_folder}/artfid_inputs" 

os.makedirs(result_folder, exist_ok=True)

mask_dir = os.path.join(result_folder, "mask")
# result_dir = "segmented/semantic_mask"
photo_in_dir = os.path.join(result_folder, "photo_mask_in")
# "photorealistc_mask_in"
photo_out_dir = os.path.join(result_folder, "photo_mask_out")
# "photorealistc_mask_out"
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(photo_in_dir, exist_ok=True)
os.makedirs(photo_out_dir, exist_ok=True)

stylized_in_dir = os.path.join(result_folder, "stylized_mask_in")
stylized_out_dir = os.path.join(result_folder, "stylized_mask_out")

os.makedirs(stylized_in_dir, exist_ok=True)
os.makedirs(stylized_out_dir, exist_ok=True)

# os.makedirs("directory/subdirectory")
# args.init_ckpt: ckpt_svox2/llff/leaves/ckpt.npz


# Loop through all the files in the folder
for filename in os.listdir(photo_folder_path):
    # Check if the current item is a file
    if os.path.isfile(os.path.join(photo_folder_path, filename)):
        img_path=os.path.join(photo_folder_path, filename)
        stylized_img_path=os.path.join(stylized_folder_path, filename)
        # print(filename)
        # print(stylized_img_path)

        base = os.path.splitext(filename)[0]

        # input_img = imageio.imread(img_path).astype(np.float32) / 255.0
        input_img = Image.open(img_path).convert('RGB')
        stylized_img = Image.open(stylized_img_path).convert('RGB')


        def decode_segmap_binary_mask(image, nc=21, select_c_list=[13]):
            label_colors = np.array([(0, 0, 0),  # 0=background
                        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                        # 11=table, 12=dog, 13=horse, 14=motorbike, 15=person
                        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

            gray = np.zeros_like(image).astype(np.uint8)
            
            for l in range(0, nc):
                if (l in select_c_list):
                    idx = image == l
                    gray[idx] = 1
            return gray

        
        # normalize_func = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_func = T.Compose([
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])

        inp = normalize_func(input_img).unsqueeze(0).to(device)

        # print('inp', inp.shape) #inp torch.Size([1, 3, 546, 980])
        

        out = net.to(device=device)(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        if semantic_type == 'room':
            if item_type == 'all':
                select_c_list = [9, 11, 20] # chair + table + tv/monitor
            if item_type == 'TV':
                select_c_list = [20]
            if item_type == 'table':
                select_c_list = [11]
            if item_type == 'chair':
                select_c_list = [9]
        else:
            select_c_list = [13] # horse
        
        semantic_img = decode_segmap_binary_mask(om, nc=21, select_c_list=select_c_list)
        # print('semantic_img min', min(min(row) for row in semantic_img))
        # print('semantic_img max', max(max(row) for row in semantic_img))
        # print('semantic_img', semantic_img.shape) #semantic_img (546, 980)
        semantic_img_orig = semantic_img

        if True:
            imageio.imwrite(
                os.path.join(mask_dir, f"{base}_seg_mask.png"),
                semantic_img.astype(np.uint8) * 255.0,
            )
        
        # Load the mask image (grayscale)
        # mask_path = "path_to_your_mask.png"
        # mask = Image.open(mask_path).convert("L")
        mask = semantic_img

        # Convert the images to numpy arrays
        image_array = np.array(input_img)
        mask_array = np.array(mask)

        stylized_img_array = np.array(stylized_img)

        # ====== photo realistic mask in ====== 
        # Apply the mask to the image
        masked_image_array = image_array * np.expand_dims(mask_array, axis=2)
        # Convert the masked image array back to PIL Image
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        # Save the masked image
        if True: 
            output_path = os.path.join(photo_in_dir, f"{base}_mask_in.png")
            masked_image.save(output_path)


         # ====== stylized mask in ====== 
        masked_image_array = stylized_img_array * np.expand_dims(mask_array, axis=2)
        # Convert the masked image array back to PIL Image
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        # Save the masked image
        if True: 
            output_path = os.path.join(stylized_in_dir, f"{base}_mask_in.png")
            masked_image.save(output_path)


        # ====== photo mask out ====== 
        # image_array = np.array(input_img)
        # mask_array = np.array(mask)
        # mask_array_revert = np.abs(np.subtract(mask_array, 1))
        mask_array_revert = np.logical_not(mask_array)
        # Apply the mask to the image
        masked_image_array = image_array * np.expand_dims(mask_array_revert, axis=2)
        # Convert the masked image array back to PIL Image
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        # Save the masked image
        if True: 
            output_path = os.path.join(photo_out_dir, f"{base}_mask_out.png")
            masked_image.save(output_path)


        # ====== stylized mask out ====== 
        masked_image_array = stylized_img_array * np.expand_dims(mask_array_revert, axis=2)
        # Convert the masked image array back to PIL Image
        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
        # Save the masked image
        if True: 
            output_path = os.path.join(stylized_out_dir, f"{base}_mask_in.png")
            masked_image.save(output_path)