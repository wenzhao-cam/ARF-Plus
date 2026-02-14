SCENE=$1
STYLE=$2
style_vgg_layer_weight='0-0-0.5-0.5-0'
style_img_scales='1-1-0.3-0.3-1'
# style_img_scales='1.5-1.5-1.5-1.8-1.5'
# style_img_scales='1.3-1.5-1-1-1'

data_type=tnt
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_scale_control_convs_weight_${style_vgg_layer_weight}_style_img_scales_${style_img_scales}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json
fi


python opt_style_scale_control_formula.py -t ${ckpt_arf} ${data_dir} \
                -c configs/tnt_fixgeom.json  \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --style_vgg_layer_weight ${style_vgg_layer_weight} \
                --style_img_scales ${style_img_scales} \
                --mse_num_epoches 1 --nnfm_num_epoches 10 \
                --content_weight 5e-3

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                --render_path
