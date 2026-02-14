SCENE=$1
STYLE1=$2
STYLE2=$3

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE1}_${STYLE2}_blend
data_dir=../data/${data_type}/${SCENE}
style_img1=../data/styles/${STYLE1}.jpg
style_img2=../data/styles/${STYLE2}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_blend_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style1 ${style_img1} \
                --style2 ${style_img2} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs_blend_style.py ${ckpt_arf} ${data_dir} \
                    --render_path
