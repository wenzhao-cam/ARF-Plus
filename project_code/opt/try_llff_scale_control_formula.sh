SCENE=$1
STYLE=$2
# scale_value=$3
# style_vgg_layer=$4 #'0-1'
style_vgg_layer_weight='0-0-0.2-0.8-0'
# scale_value=$4

# scale_value=1
# SCENE=flower
# STYLE=39
style_img_scales='1.0-1.0-1.0-0.4-1.0'

# style_img_scales='1.5-1.5-1.5-1.5-1.5'
# style_vgg_layer_weight='0.3-0.7-0-0-0'
# style_img_scales='0.5-0.5-0.5-0.5-0.5'
# style_img_scales='1.5-1.5-0.6-0.7-1.5'
# style_img_scales='0.4-0.4-0.4-0.4-0.4'
# style_img_scales='0.5-0.5-0.5-0.5-0.5'
# style_img_scales='2.5-2.5-2.5-2.5-2.5'

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_scale_control_convs_weight_${style_vgg_layer_weight}_style_img_scales_${style_img_scales}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg




if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_scale_control_formula.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --style_vgg_layer_weight ${style_vgg_layer_weight} \
                --style_img_scales ${style_img_scales} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 
                

python render_imgs.py ${ckpt_arf} ${data_dir} \
                    --render_path
