SCENE=$1
STYLE1=$2
STYLE2=$3

data_type=tnt
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE1}_${STYLE2}_combination_controls_semantic
data_dir=../data/${data_type}/${SCENE}
style_img1=../data/styles/${STYLE1}.jpg
style_img2=../data/styles/${STYLE2}.jpg

semantic_type=horse

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json
fi


python opt_style_spatial_combination_controls_semantic.py -t ${ckpt_arf} ${data_dir} \
                -c configs/tnt_fixgeom.json  \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style1 ${style_img1} \
                --style2 ${style_img2} \
                --mse_num_epoches 1 --nnfm_num_epoches 10 \
                --content_weight 5e-4 \
                --semantic_type ${semantic_type}

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                --render_path

  # --content_weight 5e-4 \