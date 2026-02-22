# -*- coding: utf-8 -*-

import csv 

def save_metrics(values, name):    
    fields = ['Accuracy', 'macro_Precision', 'macro_Recall', 'macro_f1Score', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'Inference_time','epoch_loss']  
    with open(name, 'w') as f:       
        # using csv.writer method from CSV package 
        write = csv.writer(f)       
        write.writerow(fields) 
        write.writerows(values) 
        
        
   
    
