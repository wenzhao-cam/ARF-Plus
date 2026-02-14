SCENE=$1
STYLE=$2

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}_spatial_control_semantic_segment_reverse
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg
semantic_type=room

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

python opt_style_spatial_control_semantic_reverse.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 1 --nnfm_num_epoches 5 \
                --content_weight 1e-4 \
                --semantic_type ${semantic_type}

python render_imgs.py ${ckpt_arf} ${data_dir} \
                    --render_path

                # --mse_num_epoches 1 --nnfm_num_epoches 3 \
                # --content_weight 5e-4 \
            