#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:01:40 2022

@author: negin
"""
#############################################################
# Importing from a sibling directory:
import sys
sys.path.append("..")
#############################################################

from utils.Transforms import *
from nets_STPP.semseg.deeplabv3plus import DeepLabV3Plus as Net1
import torch.nn as nn


project_name ="WetLab_rhexis"

# For binary segmentation, num_classes = 1 (the backgroung class is not considered)
num_classes =1

Framework_name = "Supervised_DeepLabV3Plus_Res50_JustInstruments"

criterion_supervised = nn.CrossEntropyLoss()#Should be used without softmax #DiceBCELoss()
criterion_SemiSupervised = ''


datasets = [
['Wetlab_rhexis_train_fold0.csv', 'Wetlab_rhexis_train_fold0.csv', 'Wetlab_rhexis_test_fold0.csv', 1],
['Wetlab_rhexis_train_fold1.csv', 'Wetlab_rhexis_train_fold1.csv', 'Wetlab_rhexis_test_fold1.csv', 1],
['Wetlab_rhexis_train_fold2.csv', 'Wetlab_rhexis_train_fold2.csv', 'Wetlab_rhexis_test_fold2.csv', 1],
['Wetlab_rhexis_train_fold3.csv', 'Wetlab_rhexis_train_fold3.csv', 'Wetlab_rhexis_test_fold3.csv', 1]
]

Learning_Rates_init = [0.005, 0.01, 0.02]
epochs = 80
batch_size = 16
size = 'determined_in_dataset'

# Warning: if the model weights are loaded, the learning rate should also change based on the number of epochs
load = False
load_path = '/storage/homefs/ng22l920/Codes/SemiSup_TMI_Results/checkpoints_Wetlab/'

load_epoch = ''#10

Results_path = '/storage/homefs/ng22l920/Codes/SemiSup_TMI_Results/'
Visualization_path = 'visualization_Wetlab/'
CSV_path = 'CSVs_Wetlab/'
Checkpoint_path = 'checkpoints_Wetlab/'
TrainIDs_path = '/storage/homefs/ng22l920/Codes/Wetlab_Phase/TrainIDs_Seg_rhexis/'

net_name = 'MixedNets'
test_per_epoch = 0.2


SemiSupervised_batch_size = 8
SemiSupervised_initial_epoch = 0

image_transforms = ''


affine_transforms = ''
affine = False                               


# Unsupervised loss-weightening function parameters:  
LW = 1

# Unsupervised average-mask weightening function parameters:
EMA_decay = 0.99

# Unsupervised average-mask weightening function parameters:
Alpha = 1

GCC = 2

strategy = ''

hard_label_thr = ''


