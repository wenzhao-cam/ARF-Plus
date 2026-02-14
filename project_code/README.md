# ARF-Plus: Explore Perceptual Flexibility in Artistic Radiance Fields for 3D Scene Stylization
Author: wl301@cam.ac.uk / wl301@cantab.ac.uk

## Quick start

### Requirement 
python 3.8.8

### Install environment
```bash
. ./create_env.sh
```
### Download data
```bash
. ./download_data.sh
```
### Stylize artistic radiance fields 

* ```llff``` represents Real Forward-Facing Scenes Dataset, ```tnt``` represents Real 360 Degree Scenes Dataset. 
* Select ```{llff/tnt/custom}``` according to your data type. For example, use ```llff``` for ```flower``` scene, ```tnt``` for ```Playground``` scene. 
* ```[style_id].jpg``` is the style image inside ```./data/styles```. For example, ```14.jpg``` is the starry night painting.
* Once all data is ready, you should find three subfolders within  ```data``` folder - ```llff```(contains forward-facing dataset), ```styles```(contains styles images), ```tnt```(contains tnt 360 degree dataset).
* Note that a photorealistic radiance field will first be reconstructed for each scene, if it doesn't exist on disk. This will take extra time.
* The photo-realistic models are generated and stored wihin ```./opt/ckpt_svox2``` and the stylized models are generated and stored within ```./opt/ckpt_arf```.


### 1. color preserve control
```bash
cd opt 
```
###### baseline

```bash
. ./try_{llff/tnt}.sh [scene_name] [style_id]

# examples
# . ./try_llff.sh trex 19;
# . ./try_tnt.sh M60 7;
```

###### color histogram matching method

```bash
. ./try_{llff/tnt}_color_control_his.sh [scene_name] [style_id]
# examples
# . ./try_llff_color_control_his.sh flower 7;
# . ./try_tnt_color_control_his.sh M60 7;

```

###### luminance-only transfer method
```bash
. ./try_{llff/tnt}_color_control_lum.sh [scene_name] [style_id]
# examples
# . ./try_llff_color_control_lum.sh flower 7;
# . ./try_tnt_color_control_lum.sh Playground 17;
```

### 2. scale control
```bash
cd opt 
```
###### baseline

```bash
. ./try_{llff/tnt}.sh [scene_name] [style_id]
# examples
# . ./try_llff.sh flower 14;
# . ./try_tnt.sh Train 6;
```

###### single style scale control 

```bash
. ./try_llff_scale_control_formula.sh [scene_name] [style_id]
# examples
# . ./try_llff_scale_control_formula.sh horns 131;
# . ./try_tnt_scale_control_formula.sh Horse 7;

# note: in try_llff_scale_control_formula.sh, 
# adjust the style_vgg_layer_weight and style_img_scales for each conv layer blocks.
```

###### multiple styles scale control with color preserve (for better visulization)

```bash
. ./try_llff_scale_control_blend_style_one_loss_func_lum_sequence_new.sh [scene_name] [style1_id] [style2_id]

. ./try_llff_scale_control_blend_style_with_color.sh [scene_name] [style1_id] [style2_id]
# examples
# . ./try_llff_scale_control_blend_style_one_loss_func_lum_sequence_new.sh flower 121 122;

# note: In nnfm_loss_scale_control_blend_styles_one_loss_func_lum_sequence_new.py, 
# 1) adjust the epoch id to determine when to initiate simultaneous transfer of coarse and fine styles （after training the coarse style separately in previous epoch). 
# 2) adjust the weights of selected convolution layer blocks for each style -  blocks_style_coarse_weight and blocks_style_fine_weight.  
# 3) adjust the size of style images styles1 and styles2. 
# 4) adjust the weights of coarse and fine styles

```


### 3. spatial control
```bash
cd opt 
```
###### baseline

```bash
# baseline - disable recolor process
. ./try_{llff/tnt}_origin_to_compare_spatial_control.sh [scene_name] [style_id]
# examples
# . ./try_llff_origin_to_compare_spatial_control.sh orchids 72;
# . ./try_tnt_origin_to_compare_spatial_control Truck 89;
```
###### single style spatial control

```bash
# depth map spatial mask
. ./try_{llff/tnt}_spatial_control.sh [scene_name] [style_id]
# examples
# . ./try_llff_spatial_control.sh leaves 46;
# . ./try_tnt_spatial_control.sh Family 9;

# inverted depth map spatial mask
. ./try_{llff/tnt}_spatial_control_reverse.sh [scene_name] [style_id]
# examples
# . ./try_llff_spatial_control_reverse.sh leaves 46;
# . ./try_tnt_spatial_control_reverse.sh Family 9;
```

```bash
# semantic segmentation map spatial mask
# 1.Room
. ./try_llff_spatial_control_semantic_segment.sh room [style_id]
# examples
# . ./try_llff_spatial_control_semantic_segment.sh room 7;
# 2.Horse
. ./try_tnt_spatial_control_semantic_segment.sh Horse [style_id]
# examples
# . ./try_tnt_spatial_control_semantic_segment.sh Horse 89;


# inverted semantic segmentation map spatial mask
# 1.Room
. ./try_llff_spatial_control_semantic_segment_reverse.sh room [style_id]
# examples
# . ./try_llff_spatial_control_semantic_segment_reverse.sh room 7;
# 2.Horse
. ./try_tnt_spatial_control_semantic_segment_reverse.sh Horse [style_id]
# examples
# . ./try_tnt_spatial_control_semantic_segment_reverse.sh Horse 89;

```

######  multiple styles spatial control 

```bash
. ./try_tnt_spatial_control_semantic_blend_styles_revert_origin_content_mask.sh [scene_name] [style1_id] [style2_id]
# examples
# . ./ try_tnt_spatial_control_semantic_blend_styles_revert_origin_content_mask.sh Horse 33 40;
# . ./ try_tnt_spatial_control_semantic_blend_styles_revert_origin_content_mask.sh Horse 40 33;
# . ./try_tnt_spatial_control_semantic_blend_styles_revert_origin_content_mask.sh Horse 100 7;
```

### 4. depth-aware control
```bash
cd opt 
```
###### baseline

```bash
. ./try_llff.sh [scene_name] [style_id]
# examples
# . ./try_llff.sh leaves 22;
```

###### ARF-Plus depth control

```bash
. ./try_llff_depth_control.sh [scene_name] [style_id]
# examples
# . ./try_llff_depth_control.sh leaves 22;
```

### 5. Combination controls
```bash
cd opt 
```
###### baseline

```bash
. ./try_{llff/tnt}.sh [scene_name] [style_id]
# examples
# . ./try_llff.sh flower 46;
# . ./try_tnt.sh Horse 131;
```

###### ARF-Plus combination control

```bash
. ./try_{llff/tnt}.sh [scene_name] [style1_id] [style2_id]
# examples
# . ./try_llff_spatial_combinations_control.sh leaves 46 26
# . ./try_tnt_spatial_combinations_control_semantic.sh Horse 131 7;
```


### Check results
The stylized artistic radiance field is inside ```opt/ckpt_arf/[scene_name]_[style_id]_[control_type]```(single style input) or ```opt/ckpt_arf/[scene_name]_[style1_id]_[style2_id]_[control_type]```(multiple styles input), while the photorealistic one is inside ```opt/ckpt_svox2/[scene_name]```.

### Custom data
Please follow the steps on [Plenoxel](https://github.com/sxyu/svox2)  to prepare your own custom data.

## Acknowledgement:
We would like to thank [Plenoxel](https://github.com/sxyu/svox2) and [ARF](https://github.com/Kai-46/ARF-svox2)authors for open-sourcing their implementations.
