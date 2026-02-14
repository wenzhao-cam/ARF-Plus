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

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap
from util import config_util

import imageio
import cv2

# from style_transfer_losses import StyleTransferLosses
from nnfm_loss_spatial_control import NNFMLoss, match_colors_for_image_set

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print('-------begin-------')
print('generate depth mask')

stylized_folder_path = "ckpt_arf/llff/flower_73_122_blend_spatial_control_old/test_renders_path"
photo_folder_path = "ckpt_svox2/llff/flower/test_renders_path"
# semantic_type = "Horse"

result_folder = "ckpt_arf/llff/flower_73_122_blend_spatial_control_old/artfid_inputs"
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

# 这里是要用从模型里render的逻辑
# data_type = 'llff'
# SCENE = 'flower'
# ckpt_dir=f'ckpt_svox2/${data_type}/${SCENE}'
# data_dir=f'../data/${data_type}/${SCENE}'


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_lpips',
                    action='store_true',
                    default=False,
                    help="Disable LPIPS (faster load)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

parser.add_argument(
    "--log_depth_map_use_thresh",
    type=float,
    default=None,
    help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term",
)

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'



if args.timing:
    args.no_lpips = True
    args.no_vid = True
    args.ray_len = False

if not args.no_lpips:
    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

# render_dir = path.join(path.dirname(args.ckpt),
#             'train_renders' if args.train else 'test_renders')
render_dir = path.join(result_folder,
            'wenzhao_test')

want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '_path'
    want_metrics = False

# Handle various image transforms
if not args.render_path:
    # Do not crop if not render_path
    args.crop = 1.0
if args.crop != 1.0:
    render_dir += f'_crop{args.crop}'
if args.ray_len:
    render_dir += f'_raylen'
    want_metrics = False

print('args.data_dir', args.data_dir)
dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_dir += '_nofg'

    # DEBUG
    #  grid.links.data[grid.links.size(0)//2:] = -1
    #  render_dir += "_chopx2"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)
    # DEBUGGING
    #  rad = [1.496031746031746, 1.6613756613756614, 1.0]
    #  half_sz = [grid.links.size(0) // 2, grid.links.size(1) // 2]
    #  pad_size_x = int(half_sz[0] - half_sz[0] / 1.496031746031746)
    #  pad_size_y = int(half_sz[1] - half_sz[1] / 1.6613756613756614)
    #  print(pad_size_x, pad_size_y)
    #  grid.links[:pad_size_x] = -1
    #  grid.links[-pad_size_x:] = -1
    #  grid.links[:, :pad_size_y] = -1
    #  grid.links[:, -pad_size_y:] = -1
    #  grid.links[:, :, -8:] = -1

    #  LAYER = -16
    #  grid.links[:, :, :LAYER] = -1
    #  grid.links[:, :, LAYER+1:] = -1

    frames = []
    #  im_gt_all = dset.gt.to(device=device)

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)

        depth_img = grid.volume_render_depth_image(cam,
                args.log_depth_map_use_thresh if
                args.log_depth_map_use_thresh else None
            )
            
        if False:
            imageio.imwrite(
                os.path.join(args.train_dir, f"origin_depth_map_{img_id:04d}_{epoch_id}_final.png"),
                depth_img.detach().cpu().numpy()
            )    

        # print('depth_img.shape', depth_img.shape)
        # print('depth_img', depth_img)

        # depth_img = viridis_cmap(depth_img.cpu())

        # depth_img = (depth_img * 255).astype("uint8")

        depth_map = depth_img.detach().cpu().numpy()

        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        # print('min_depth', min_depth)
        # print('max_depth', max_depth)

        normalized_depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        normalized_depth_map = normalized_depth_map * 255.0

        # print('normalized_depth_map.shape', normalized_depth_map.shape)

        otsu_threshold, image_result = cv2.threshold(
            normalized_depth_map.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # print("Obtained threshold: ", otsu_threshold)

        if True:
            imageio.imwrite(
                os.path.join(mask_dir, f"{img_id:04d}_seg_mask.png"),
                image_result.astype(np.uint8),
            )
    
        print('image_result', image_result.shape)
        filename = f"{img_id:04d}.png"
        # need to read from photo-realistic and stylized to find the corresponding file
        # then apply masks
        
        img_path=os.path.join(photo_folder_path, filename)
        stylized_img_path=os.path.join(stylized_folder_path, filename)
        # print(filename)
        # print(stylized_img_path)

        base = os.path.splitext(filename)[0]

        # input_img = imageio.imread(img_path).astype(np.float32) / 255.0
        input_img = Image.open(img_path).convert('RGB')
        stylized_img = Image.open(stylized_img_path).convert('RGB')

        mask = image_result

        # Convert the images to numpy arrays
        image_array = np.array(input_img)
        mask_array = np.array(mask)
        mask_array = np.logical_not(np.logical_not(mask_array))

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






        
    #     im = grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
    #     if args.ray_len:
    #         minv, meanv, maxv = im.min().item(), im.mean().item(), im.max().item()
    #         im = viridis_cmap(im.cpu().numpy())
    #         cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
    #                     0, 0.5, [255, 0, 0])
    #         im = torch.from_numpy(im).to(device=device)
    #     im.clamp_(0.0, 1.0)

    #     if not args.render_path:
    #         im_gt = dset.gt[img_id].to(device=device)
    #         mse = (im - im_gt) ** 2
    #         mse_num : float = mse.mean().item()
    #         psnr = -10.0 * math.log10(mse_num)
    #         avg_psnr += psnr
    #         if not args.timing:
    #             ssim = compute_ssim(im_gt, im).item()
    #             avg_ssim += ssim
    #             if not args.no_lpips:
    #                 lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
    #                         im.permute([2, 0, 1]).contiguous(), normalize=True).item()
    #                 avg_lpips += lpips_i
    #                 print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
    #             else:
    #                 print(img_id, 'PSNR', psnr, 'SSIM', ssim)
    #     img_path = path.join(render_dir, f'{img_id:04d}.png');
    #     im = im.cpu().numpy()
    #     if not args.render_path:
    #         im_gt = dset.gt[img_id].numpy()
    #         im = np.concatenate([im_gt, im], axis=1)
    #     if not args.timing:
    #         im = (im * 255).astype(np.uint8)
    #         if not args.no_imsave:
    #             imageio.imwrite(img_path,im)
    #         if not args.no_vid:
    #             frames.append(im)
    #     im = None
    #     n_images_gen += 1
    # if want_metrics:
    #     print('AVERAGES')

    #     avg_psnr /= n_images_gen
    #     with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
    #         f.write(str(avg_psnr))
    #     print('PSNR:', avg_psnr)
    #     if not args.timing:
    #         avg_ssim /= n_images_gen
    #         print('SSIM:', avg_ssim)
    #         with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
    #             f.write(str(avg_ssim))
    #         if not args.no_lpips:
    #             avg_lpips /= n_images_gen
    #             print('LPIPS:', avg_lpips)
    #             with open(path.join(render_dir, 'lpips.txt'), 'w') as f:
    #                 f.write(str(avg_lpips))
    # if not args.no_vid and len(frames):
    #     vid_path = render_dir + '.mp4'
    #     imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg




# # Loop through all the files in the folder
# for filename in os.listdir(photo_folder_path):
#     # Check if the current item is a file
#     if os.path.isfile(os.path.join(photo_folder_path, filename)):
#         img_path=os.path.join(photo_folder_path, filename)
#         stylized_img_path=os.path.join(stylized_folder_path, filename)
#         # print(filename)
#         # print(stylized_img_path)

#         base = os.path.splitext(filename)[0]

#         # input_img = imageio.imread(img_path).astype(np.float32) / 255.0
#         input_img = Image.open(img_path).convert('RGB')
#         stylized_img = Image.open(stylized_img_path).convert('RGB')


#         def decode_segmap_binary_mask(image, nc=21, select_c_list=[13]):
#             label_colors = np.array([(0, 0, 0),  # 0=background
#                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
#                         (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
#                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
#                         (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
#                         # 11=table, 12=dog, 13=horse, 14=motorbike, 15=person
#                         (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
#                         # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
#                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

#             gray = np.zeros_like(image).astype(np.uint8)
            
#             for l in range(0, nc):
#                 if (l in select_c_list):
#                     idx = image == l
#                     gray[idx] = 1
#             return gray

        
#         # normalize_func = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         normalize_func = T.Compose([
#                  T.ToTensor(), 
#                  T.Normalize(mean = [0.485, 0.456, 0.406], 
#                              std = [0.229, 0.224, 0.225])])

#         inp = normalize_func(input_img).unsqueeze(0).to(device)

#         # print('inp', inp.shape) #inp torch.Size([1, 3, 546, 980])
        

#         out = net.to(device=device)(inp)['out']
#         om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
#         if semantic_type == 'room':
#             select_c_list = [9, 11, 20] # chair + table + tv/monitor
#             select_c_list = [20] # chair + table + tv/monitor
#         else:
#             select_c_list = [13] # horse
        
#         semantic_img = decode_segmap_binary_mask(om, nc=21, select_c_list=select_c_list)
#         # print('semantic_img min', min(min(row) for row in semantic_img))
#         # print('semantic_img max', max(max(row) for row in semantic_img))
#         # print('semantic_img', semantic_img.shape) #semantic_img (546, 980)
#         semantic_img_orig = semantic_img

#         if True:
#             imageio.imwrite(
#                 os.path.join(mask_dir, f"{base}_seg_mask.png"),
#                 semantic_img.astype(np.uint8) * 255.0,
#             )
        
#         # Load the mask image (grayscale)
#         # mask_path = "path_to_your_mask.png"
#         # mask = Image.open(mask_path).convert("L")
#         mask = semantic_img

#         # Convert the images to numpy arrays
#         image_array = np.array(input_img)
#         mask_array = np.array(mask)

#         stylized_img_array = np.array(stylized_img)

#         # ====== photo realistic mask in ====== 
#         # Apply the mask to the image
#         masked_image_array = image_array * np.expand_dims(mask_array, axis=2)
#         # Convert the masked image array back to PIL Image
#         masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
#         # Save the masked image
#         if True: 
#             output_path = os.path.join(photo_in_dir, f"{base}_mask_in.png")
#             masked_image.save(output_path)


#          # ====== stylized mask in ====== 
#         masked_image_array = stylized_img_array * np.expand_dims(mask_array, axis=2)
#         # Convert the masked image array back to PIL Image
#         masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
#         # Save the masked image
#         if True: 
#             output_path = os.path.join(stylized_in_dir, f"{base}_mask_in.png")
#             masked_image.save(output_path)


#         # ====== photo mask out ====== 
#         # image_array = np.array(input_img)
#         # mask_array = np.array(mask)
#         # mask_array_revert = np.abs(np.subtract(mask_array, 1))
#         mask_array_revert = np.logical_not(mask_array)
#         # Apply the mask to the image
#         masked_image_array = image_array * np.expand_dims(mask_array_revert, axis=2)
#         # Convert the masked image array back to PIL Image
#         masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
#         # Save the masked image
#         if True: 
#             output_path = os.path.join(photo_out_dir, f"{base}_mask_out.png")
#             masked_image.save(output_path)


#         # ====== stylized mask out ====== 
#         masked_image_array = stylized_img_array * np.expand_dims(mask_array_revert, axis=2)
#         # Convert the masked image array back to PIL Image
#         masked_image = Image.fromarray(masked_image_array.astype(np.uint8))
#         # Save the masked image
#         if True: 
#             output_path = os.path.join(stylized_out_dir, f"{base}_mask_in.png")
#             masked_image.save(output_path)


