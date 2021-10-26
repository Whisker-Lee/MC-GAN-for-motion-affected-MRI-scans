#!/bin/bash
#SBATCH --mail-user=sli339@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=./log/test/vgg1.stdout
#SBATCH --error=./log/test/vgg1.stderr
#SBATCH --nodes=1
#SBATCH --exclude=node[003]
#SBATCH --gres=gpu:1    # GPU request
#SBATCH --mem=160gb       # Job memory request 




module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0
module load openmpi/cuda/64/3.1.4
module load cuda10.1/toolkit/10.1.243
module load opencv3-py36-cuda10.1-gcc/3.4.10 
module load tensorflow-py36-cuda10.1-gcc/1.15.3

#pip3 install opencv-python --user
#pip3 install matplotlib --user
#pip3 install imutils --user
#pip3 install scikit-image==0.16.2 --user

python3.6 main_abide2_50.py  --mode test_only --pre_trained_model ./model/DeblurrGAN-89 --num_of_down_scale 4 --gen_resblocks 16 --discrim_blocks 5 --decay_step 500 --channel 3
python3.6 main_abide3_50.py  --mode test_only --pre_trained_model ./model/DeblurrGAN-89 --num_of_down_scale 4 --gen_resblocks 16 --discrim_blocks 5 --decay_step 500 --channel 3
python3.6 main_abide_50.py  --mode test_only --pre_trained_model ./model/DeblurrGAN-89 --num_of_down_scale 4 --gen_resblocks 16 --discrim_blocks 5 --decay_step 500 --channel 3