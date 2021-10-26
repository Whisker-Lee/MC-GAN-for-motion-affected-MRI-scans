#!/bin/bash
#SBATCH --mail-user=sli339@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./log/train/vgg1.stdout
#SBATCH --error=./log/train/vgg1.stderr
#SBATCH --nodes=1
#SBATCH --exclude=node[001,002]
#SBATCH --gres=gpu:1    # GPU request
#SBATCH --mem=160gb       # Job memory request


module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0
module load openmpi/cuda/64/3.1.4
module load cuda10.1/toolkit/10.1.243
module load opencv3-py36-cuda10.1-gcc/3.4.10 
module load tensorflow-py36-cuda10.1-gcc/1.15.3

pip3 install opencv-python --user
pip3 install matplotlib --user
pip3 install imutils --user


# python3.6 modify_data.py
#python3.6 dataSetGenerator.py  --path /home/shared/shangjin_oasis_new/x/oasis --SaveTo /home/shared/shangjin_oasis_new/ae_DataSets/x_DataSets

#python3.6 train_vgg19.py --dataset oasis --batch 16 --epochs 20

#python3.6 dataSetGenerator.py  --path /home/shared/shangjin_oasis_new/x/oasis --SaveTo /home/shared/shangjin_oasis_new/ae_align_DataSets/x_DataSets

python3.6 main.py --num_of_down_scale 4 --gen_resblocks 16 --discrim_blocks 5 --decay_step 500 --pre_trained_model ./model/DeblurrGAN-0  --fine_tuning True
