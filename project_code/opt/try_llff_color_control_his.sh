SCENE=$1
STYLE=$2

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_color_control_hist
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg

echo "ckpt_svox2: $ckpt_svox2"
echo "ckpt_arf: $ckpt_arf"
echo "data_dir: $data_dir"
echo "style_img: $style_img"

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_preserve_color.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 --disable_recolor 1

# python render_imgs_preserve_color.py ${ckpt_arf} ${data_dir} \
#                     --render_path
                # --mse_num_epoches 2 --nnfm_num_epoches 5 \

python render_imgs_preserve_color.py ${ckpt_arf} ${data_dir} \
                    --render_path