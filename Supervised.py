from __future__ import print_function
import random 
import argparse
import logging 
import os 

import csv

import sys
sys.path.append("../")
import numpy as np 
import torch 
import torch.nn as nn 
from torch import optim 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import importlib
import argparse
import wandb



from utils.TrainUtils import create_directory
from utils.import_helper import import_config
from utils.dataset_phase_aug import BasicDataset as BasicDataset_CSV
from utils.dataset_phase_test import BasicDataset as BasicDataset_CSV_test
from utils.eval_metrics import eval_metrics
from utils.save_metrics import save_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)



#from utils_SemiSup.DataLoaders_STPP import DataLoader_SemiSup


class printer(nn.Module):
        def __init__(self, global_dict=globals()):
            super(printer, self).__init__()
            self.global_dict = global_dict
            self.except_list = []
        def debug(self,expression):
            frame = sys._getframe(1)

            print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))

        def namestr(self,obj, namespace):
            return [name for name in namespace if namespace[name] is obj]     
        
        def forward(self,x):
            for i in x:
                if i not in self.except_list:
                    name = self.namestr(i, globals())
                    if len(name)>1:
                        self.except_list.append(i)
                        for j in range(len(name)):
                            self.debug(name[j])
                    else:  
                        self.debug(name[0])

def train_net(net,
              epochs=30,
              batch_size=16,
              lr=0.001,
              device='cuda',
              save_cp=True
              ):

    TESTS = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)  # Ensure the model is on GPU
  
    train_dataset = BasicDataset_CSV(dir_train)
    test_dataset = BasicDataset_CSV_test(dir_test, batch_size=batch_size)

    n_train = len(train_dataset)
    print(f'n_train:{n_train}')
   
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=False, drop_last=True)

    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=False, drop_last=True)
    n_test = len(test_dataset)

    print(f'n_test: {n_test}')


    total_iters = len(train_loader) * epochs*epoch_multiplier
    inference_step = np.floor(np.ceil(n_train)/test_per_epoch)*epoch_multiplier
    print(f'Inference Step:{inference_step}')

    
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Test size:       {n_test}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD([{'params': net.feature_extractor.parameters(), 'lr': lr/10},
                     {'params': [param for name, param in net.named_parameters()
                                 if 'feature_extractor' not in name],
                      'lr': lr}], lr=lr, momentum=0.9, weight_decay=1e-4)
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.9)

    criterion =  nn.CrossEntropyLoss() 
    #criterion = nn.BCELoss()
    test_counter = 1
    for epoch in range(epochs*epoch_multiplier):
        net.train()
        
        print(f'Epoch {epoch}')
        epoch_loss = 0
        with tqdm(total=n_train*batch_size, desc=f'Epoch {epoch + 1}/{epochs*epoch_multiplier}', unit='clips') as pbar:
            for batch in train_loader:

                

                vids = batch['video'].squeeze()
                true_labels = batch['label'].squeeze()
                

                labels_pred = net(vids).permute(0,2,1) # for CE
                #labels_pred = net(vids) # for BCE

                loss_main = criterion(labels_pred, true_labels)
                loss_wandb = loss_main
                
                loss = loss_main
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
            
                (loss_main).backward()
                optimizer.step()

                pbar.update(vids.shape[0])
                global_step += 1
                
                ########################################################################
                lr1 = lr * (1 - global_step / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr1/10
                optimizer.param_groups[1]["lr"] = lr1
                ########################################################################
                if (global_step) % (inference_step) == 0: # Should be changed if the condition that the n_train%batch_size != 0
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        
                
                    accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time = eval_metrics(net, test_loader, device, test_counter, save_test, num_classes=num_classes, save=False)
                    print(f'Accuracy:{accuracy}')

                    print(f'macro_recall:{macro_recall}')
                    print(f'macro_precision:{macro_precision}')
                    print(f'marco_f1score:{macro_f1}')

                    print(f'weighted_recall:{weighted_recall}')
                    print(f'weighted_precision:{weighted_precision}')
                    print(f'weighted_f1:{weighted_f1}')
                    print (f'confusion_matrix:{confusion_matrix}')
                
                    TESTS.append([accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time, epoch_loss])
                    

                    test_counter = test_counter+1
                        
                    wandb.log({'Train_Loss': loss_wandb,
                            'Test_Accuracy': accuracy,
                            'Test_f1score': weighted_f1,
                            })
                    
       

        #scheduler.step()

        if save_cp and ((epoch + 1) in [epochs // 3, epochs * 2 // 3, epochs]):
        #if save_cp and (epoch + 1) == epochs*epoch_multiplier:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            torch.save(net.state_dict(),
            dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')    
           
    accuracy, macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, inference_time = eval_metrics(net, test_loader, device, test_counter, save_test, num_classes=num_classes, save=False)
    save_metrics(TESTS, csv_name)
     

if __name__ == '__main__':
    args = parser.parse_args()
    config_file = args.config
    my_conf = importlib.import_module(config_file)
    criterion_supervised, criterion_SemiSupervised, datasets, Framework_name, num_classes, \
    Learning_Rates_init, epochs, batch_size, size, Results_path, Visualization_path, \
    CSV_path, project_name, load, load_path, net_name, test_per_epoch, Checkpoint_path, Net1, \
    hard_label_thr, SemiSupervised_batch_size, SemiSupervised_initial_epoch, image_transforms, \
    affine, affine_transforms, LW, EMA_decay, Alpha, strategy, GCC, TrainIDs_path = import_config.execute(my_conf)


    print("inside main")
    print(f'Cuda Availability: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device : {device}')
    print(f'Cuda Device Number: {torch.cuda.current_device()}')
    print(f'Cuda Device Name: {torch.cuda.get_device_name(0)}')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    printer1 = printer()       
    
    print('CONFIGS:________________________________________________________')
        
    printer1 = printer()       
    
    
    try:
        for LR in range(len(Learning_Rates_init)):
            run = wandb.init(project=project_name+'_'+ net_name+'_'+str(Learning_Rates_init[LR])+'_'+str(batch_size), entity="negin_gh",
                name=Framework_name+'_'+ net_name+"_init_epoch_"+ str(SemiSupervised_initial_epoch)+ '_'+datasets[0][0][:-6] +'_'+strategy+"_GCC_"+ str(GCC),
                reinit=True)
            wandb.config = {
                "learning_rate": Learning_Rates_init[LR],
                "epochs": epochs,
                "batch_size": batch_size,
                "net_name": net_name,
                "semi_supervised_batch_size": SemiSupervised_batch_size,
                "semi_supervised_initial_epoch": SemiSupervised_initial_epoch,
                "Dataset": datasets[0][0],

                }
            for c in range(len(datasets)):      
            

                print(f'Initializing the learning rate: {Learning_Rates_init[LR]}')

                dataset_name = datasets[c][0]
                epoch_multiplier = datasets[c][-1]
                epoch_multiplier = 1
                
                

                save_test = Results_path + Visualization_path + Framework_name +'_'+ dataset_name + '_'+ net_name + '_BS_' + str(batch_size) + "_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSupervised_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_Affine_'+str(affine)+'/'
                
                dir_checkpoint = Results_path + Checkpoint_path +Framework_name +'_'+ dataset_name + '_'+ net_name + '_BS_' + str(batch_size) +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSupervised_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_Affine_'+str(affine)+'/'
                csv_name = Results_path + CSV_path +Framework_name +'_'+ dataset_name + '_' +net_name + '_BS_' + str(batch_size) +"_GCC_"+ str(GCC) +'_init_epoch_'+str(SemiSupervised_initial_epoch)+'_'+str(Learning_Rates_init[LR])+'_Affine_'+str(affine)+'.csv'
                
                create_directory(Results_path + Visualization_path)
                create_directory(Results_path + Checkpoint_path)
                create_directory(Results_path + CSV_path)


                load_path1 = ''
                
                dir_train  = TrainIDs_path + datasets[c][0]
                
                dir_test = TrainIDs_path + datasets[c][1]


                net = Net1
                logging.info(f'Network:\n'
                             f'\t{net.n_classes} output channels (classes)\n')

                
                net.to(device=device)
                
                
                train_net(net=net,
                          epochs=epochs,
                          batch_size=batch_size,
                          lr=Learning_Rates_init[LR],
                          device=device)
            
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)