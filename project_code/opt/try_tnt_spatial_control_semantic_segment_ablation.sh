SCENE=$1
STYLE=$2

data_type=tnt
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_ablation_new_new_spatial_control_semantic_segment
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg
semantic_type=horse

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json
fi


python opt_style_spatial_control_semantic_ablation.py -t ${ckpt_arf} ${data_dir} \
                -c configs/tnt_fixgeom.json  \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 1 --nnfm_num_epoches 7 \
                --content_weight 5e-4 \
                --semantic_type ${semantic_type}

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                --render_path

  # --content_weight 5e-4 \