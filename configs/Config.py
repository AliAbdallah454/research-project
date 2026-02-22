#############################################################
# Importing from a sibling directory:
import sys
sys.path.append("..")
#############################################################


from nets.VGG_Transformer import CNN_Transformer as Net
import torch.nn as nn

Net1 = Net(10,1024)

datasets = [
# ['train_fold_1.csv', 'test_fold_1_sequences.csv', 1],
# ['train_fold_2.csv', 'test_fold_2_sequences.csv', 1],
['train_fold_3.csv', 'test_fold_3_sequences.csv', 1],
# ['train_fold_4.csv', 'test_fold_4_sequences.csv', 1]
]

Learning_Rates_init = [0.005]
epochs = 60
batch_size = 16
size = (256, 256)

Dataset_Path_Train = ''
Dataset_Path_SemiTrain = ''
Dataset_Path_Test = ''
mask_folder = ''
Results_path = '/storage/homefs/ng22l920/Codes/Wetlab_Results/'
Visualization_path = 'visualization/'
Checkpoint_path = 'checkpoints/'
CSV_path = 'CSVs/'
TrainIDs_path = 'TrainIDs_Phase/'
error_path = 'code_errors/'
project_name = "Wetlab_aug"
criterion_supervised = nn.CrossEntropyLoss()#Should be used without softmax #DiceBCELoss()
criterion_SemiSupervised = ''

hard_label_thr = ''

# Warning: if the model weights are loaded, the learning rate should also change based on the number of epochs
load = False
load_path = ''
load_epoch = ''

net_name = 'VGG_Transformer'
Framework_name = "Supervised" 
test_per_epoch = 0.2
num_classes = 3

ensemble_batch_size = ''
SemiSup_initial_epoch = ''
supervised_share = ''
SemiSupervised_batch_size = ''
SemiSupervised_initial_epoch = ''
strategy = ''

image_transforms = ''

affine = ''
affine_transforms = ''

# Unsupervised loss-weightening function parameters:  
LW = 1
GCC = 2
Alpha = 1

# Unsupervised average-mask weightening function parameters:
EMA_decay = 0.99