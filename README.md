# DPCT
DPCT: Disentangling Personal Style and Content for Chinese Handwriting Generation with Transformer

Machine Vision and Learning Lab, Computer Science and Information Engineering, National Chung Cheng University
國立中正大學 資訊工程學系 機器視覺學習實驗室

It's my master's thesis and implementation code.

## Introduction
Based on the implementation of [SDT]( https://github.com/dailenson/SDT ), the training process is modified, and the content encoder is used to extract potential content features of style samples for other subsequent comparative learning applications.

Thanks for their great work !!

## Get Started
### Git Clone
git clone https://github.com/AUDOSt0ck1ng/DPCT.git

### Create Conda Env
conda create --name DPCT python=3.8
y
conda activate DPCT

### Install dependencies
cd ./DPCT
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
y
pip3 install -r requirements.txt

### Prepare Data, Use the Offered Data from [SDT]( https://github.com/dailenson/SDT ): Training, Testing, and 
ln -s "your_decompression_path" "./data"

