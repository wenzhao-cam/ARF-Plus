SCENE=$1
STYLE1=$2
STYLE2=$3

# scale_value=$3
# style_vgg_layer=$4 #'0-1'
style1_vgg_layer_weight=$4 #'0-0-0.5-0.5-0'
style2_vgg_layer_weight=$5 #'0-0-0.5-0.5-0'
# scale_value=$4

# scale_value=1
# SCENE=flower
# STYLE=39
# style_vgg_layer_weight='0.3-0.7-0-0-0'
# style_img_scales='0.5-0.5-0.5-0.5-0.5'
style1_img_scales='1-1-1-1-1'
style2_img_scales='1-1-1-1-1'
# style_img_scales='0.4-0.4-0.4-0.4-0.4'
# style_img_scales='0.5-0.5-0.5-0.5-0.5'
# style_img_scales='2.5-2.5-2.5-2.5-2.5'

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE1}_${STYLE2}_scale_control_weight1_${style1_vgg_layer_weight}_weight2_${style2_vgg_layer_weight}
data_dir=../data/${data_type}/${SCENE}
style_img1=../data/styles/${STYLE1}.jpg
style_img2=../data/styles/${STYLE2}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_scale_control_formula.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style1 ${style_img1} \
                --style2 ${style_img2} \
                --style1_vgg_layer_weight ${style1_vgg_layer_weight} \
                --style2_vgg_layer_weight ${style2_vgg_layer_weight} \
                --style1_img_scales ${style1_img_scales} \
                --style2_img_scales ${style2_img_scales} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 
                

python render_imgs.py ${ckpt_arf} ${data_dir} \
                    --render_path
