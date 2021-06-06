#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -o output/finetuning/o.$JOB_ID

source /gs/hs0/tga-i/sugiyama.y.al/TIMM/TIMM_386/bin/activate
. /etc/profile.d/modules.sh
module load cuda/11.0.194 cudnn/8.1

echo '--Start--'
echo `date`

export NUM_PROC=4
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC train.py ./ \
    --pretrained \
    --pretrained-path /gs/hs0/tga-i/sugiyama.y.al/TIMM/pytorch-image-models/train_result/PreTraining_vit_deit_tiny_patch16_224_1k/checkpoint-160.pth.tar \
    --dataset CIFAR10 \
    --num-classes 10 \
    --model vit_deit_tiny_patch16_224 \
    --input-size 3 224 224 \
    --opt sgd \
    --batch-size 192 \
    --epochs 1000 \
    --cooldown-epochs 0 \
    --lr 0.01 \
    --sched cosine \
    --warmup-epochs 5 \
    --weight-decay 0.0001 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated-aug \
    --mixup 0.8 \
    --cutmix 1.0 \
    --log-wandb \
    --output train_result \
    --experiment Finetuning_vit_deit_tiny_patch16_224_imnet1k_to_CIFAR10 \
    --id_wandb Finetuning_vit_deit_tiny_patch16_224_imnet1k_to_CIFAR10_v1 \
    -j 4

echo '--End--'
echo `date`