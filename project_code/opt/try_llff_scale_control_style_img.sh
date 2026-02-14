SCENE=$1
STYLE=$2
scale_value=$3
style_vgg_layer=$4 #'0-1'
# scale_value=$4

# scale_value=0.5
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_scale_control_convs${style_vgg_layer}_resize${scale_value}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg




if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_scale_control_style_img.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --scale_value ${scale_value} \
                --style_vgg_layer ${style_vgg_layer} \
                --mse_num_epoches 1 --nnfm_num_epoches 4 \
                --content_weight 1e-3 
                

# python render_imgs.py ${ckpt_arf} ${data_dir} \
#                     --render_path --no_imsave
