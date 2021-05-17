#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=02:00:00
#$ -j y
#$ -o output/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/TIMM/TIMM_386/bin/activate
. /etc/profile.d/modules.sh
module load cuda/11.0.194 cudnn/8.1

echo 'Hello World'

export NUM_PROC=4
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py /gs/hs0/tga-i/sugiyama.y.al/datasets/ILSVRC2012/fakeimages_v1 \
    --model vit_deit_tiny_patch16_224 \
    --opt adamw \
    --batch-size 128 \
    --epochs 1 \
    --cooldown-epochs 0 \
    --lr 0.1 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.05 \
    --smoothing 0.1 \
    --drop-path 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --reprob 0.25 \
    --log-wandb \
    --output train_result \
    --experiment Test_PreTraining_vit_deit_tiny_patch16_224_fake_1k \
    -j 4

    echo 'Hello World'