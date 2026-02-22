#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:08:36 2026

@author: djayadeep
"""
import torch
from torchmetrics.functional import accuracy, f1_score, recall, precision
from torchmetrics.functional.classification import multiclass_confusion_matrix
import os
import timeit

def eval_metrics(net, loader, device, test_counter, save_dir, num_classes, rep_head=False, save=True):
    net.eval()
    Inference_time = []
    label_pred_list = []
    true_labels_list = []

    try:
       if not os.path.exists(save_dir):
           os.mkdir(save_dir)
    except OSError:
       pass        

    for batch in loader:
        # 1. Get data and move to GPU
        try:
            vids, true_labels = batch['video'], batch['label']
        except KeyError:
            vids, true_labels = batch['video_teacher'], batch['label']
        
        # Ensure data is on the same device as the model
        vids = vids.to(device)
        true_labels = true_labels.to(device)

        start = timeit.default_timer()            
        with torch.no_grad():
            if rep_head:
                labels_pred, _ = net(vids)
            else:
                labels_pred = net(vids) 
            
        stop = timeit.default_timer()
        Inference_time.append(stop - start)

        label_pred_list.append(labels_pred.permute(0, 2, 3, 1).reshape(-1, num_classes))
        true_labels_list.append(true_labels.reshape(-1))

    full_label_pred = torch.cat(label_pred_list, dim=0)
    full_true_labels = torch.cat(true_labels_list, dim=0)

    mask = full_true_labels != 255
    full_label_pred = full_label_pred[mask]
    full_true_labels = full_true_labels[mask]

    Accuracy = float(accuracy(full_label_pred, full_true_labels, task="multiclass", num_classes=num_classes))
    macro_Precision = float(precision(full_label_pred, full_true_labels, average='macro', task="multiclass", num_classes=num_classes))
    macro_Recall = float(recall(full_label_pred, full_true_labels, average='macro', task="multiclass", num_classes=num_classes))
    macro_f1Score = float(f1_score(full_label_pred, full_true_labels, average='macro', task="multiclass", num_classes=num_classes))

    weighted_precision = float(precision(full_label_pred, full_true_labels, average='weighted', task="multiclass", num_classes=num_classes))
    weighted_recall = float(recall(full_label_pred, full_true_labels, average='weighted', task="multiclass", num_classes=num_classes))
    weighted_f1 = float(f1_score(full_label_pred, full_true_labels, average='weighted', task="multiclass", num_classes=num_classes))

    confusion_matrix = multiclass_confusion_matrix(full_label_pred, full_true_labels, num_classes=num_classes)

    net.train()
    return Accuracy, macro_Precision, macro_Recall, macro_f1Score, weighted_precision, weighted_recall, weighted_f1, confusion_matrix, sum(Inference_time)/len(Inference_time)
