import os
import parser
import argparse

import pandas as pd
import numpy as np
from captum.attr import *
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

from learn.model import CiteAutoencoder_CITEseq, CiteAutoencoder_SHAREseq, CiteAutoencoder_TEAseq
from learn.train import train_model
from util import setup_seed, MyDataset,ToTensor, read_h5_data, read_fs_label, get_vae_simulated_data_from_sampling, get_encodings

parser = argparse.ArgumentParser("Matilda")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--augmentation', type=bool, default= True, help='if augmentation or not')
parser.add_argument('--fs', type=bool, default= True, help='if doing feature selection or not')
parser.add_argument('--save_latent_space', type=bool, default= True, help='save latent space')
parser.add_argument('--save_simulated_result', type=bool, default= False, help='save simulation result')

############# for data build ##############
parser.add_argument('--nfeatures_rna', type=int, default=11062, help='number of RNA features')
parser.add_argument('--nfeatures_adt', type=int, default=0, help='number of ADT features')
parser.add_argument('--nfeatures_atac', type=int, default=0, help='number of ATAC features')
parser.add_argument('--classify_dim', type=int, default=26, help='the number of cell types')

##############  for training #################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--lr', type=float, default=0.02, help='init learning rate')

############# for model build ##############
parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')
parser.add_argument('--hidden_adt', type=int, default=30, help='the number of neurons for ADT layer')
parser.add_argument('--hidden_atac', type=int, default=185, help='the number of neurons for ATAC layer')

args = parser.parse_args()
setup_seed(args.seed) ### set random seed in order to reproduce the result
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if args.nfeatures_adt > 0 and args.nfeatures_atac == 0:
    mode = "CITEseq"
    feature_num = args.nfeatures_rna + args.nfeatures_adt 
if args.nfeatures_adt == 0 and args.nfeatures_atac > 0:
    mode = "SHAREseq"
    feature_num = args.nfeatures_rna + args.nfeatures_atac 
if args.nfeatures_adt > 0 and args.nfeatures_atac > 0:
    mode = "TEAseq"
    feature_num = args.nfeatures_rna + args.nfeatures_adt + args.nfeatures_atac 
print("The dataset is", mode)    
output_v = []
acc1_v = []
acc2_v = []

train_data_path = "../data/{}/train.h5".format(mode)      
train_label_path = "../data/{}/train_cty.csv".format(mode)   
test_data_path = "../data/{}/test.h5".format(mode)     
test_label_path = "../data/{}/test_cty.csv".format(mode)   
model_save_path = "result/marker/{}/".format(mode)    
save_fs_eachcell = "result/marker/{}/".format(mode)   

########load and deal with dataset########
train_data = read_h5_data(train_data_path)
train_label = read_fs_label(train_label_path)
train_transformed_dataset = MyDataset(train_data, train_label)
train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)

test_data = read_h5_data(test_data_path)
test_label = read_fs_label(test_label_path)
test_transformed_dataset = MyDataset(test_data, test_label)
test_dl = DataLoader(test_transformed_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)

#######build model#########
if mode == "CITEseq":
	model = CiteAutoencoder_CITEseq(args.nfeatures_rna, args.nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, args.classify_dim)
elif mode == "SHAREseq":
	model = CiteAutoencoder_SHAREseq(args.nfeatures_rna, args.nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, args.classify_dim)
elif mode == "TEAseq":
	model = CiteAutoencoder_TEAseq(args.nfeatures_rna, args.nfeatures_adt, args.nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, args.classify_dim)

#model = nn.DataParallel(model).cuda() #multi gpu
model = model.cuda() #one gpu

########train model#########
model, acc1, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=args.epochs, classify_dim = args.classify_dim, best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)
average1 = torch.mean(torch.Tensor(acc1))

##################prepare to do augmentation##################            


if args.augmentation == True:
    stage1_list = []
    for i in np.arange(0, args.classify_dim):
        stage1_list.append([i, train_num[i]])
        stage1_df = pd.DataFrame(stage1_list)
    if args.classify_dim%2==0:
        train_median = np.sort(train_num)[int(args.classify_dim/2)-1]
    else: 
        train_median = np.median(train_num)
    median_anchor = stage1_df[stage1_df[1] == train_median][0]
    train_major = stage1_df[stage1_df[1] > train_median]
    train_minor = stage1_df[stage1_df[1] < train_median]
    anchor_fold = np.array((train_median)/(train_minor[:][1]))
    minor_anchor_cts = train_minor[0].to_numpy()
    major_anchor_cts = train_major[0].to_numpy()

    index = (train_label == int(np.array(median_anchor))).nonzero(as_tuple=True)[0]
    anchor_data = train_data[index.tolist(),:]
    anchor_label = train_label[index.tolist()]
    new_data = anchor_data 
    new_label = anchor_label

    ##############random downsample major cell types##############
    j=0
    for anchor in major_anchor_cts:     
        anchor_num = np.array(train_major[1])[j]
        N = range(anchor_num)
        ds_index = random.sample(N,int(train_median))
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_data = anchor_data[ds_index,:]
        anchor_label = anchor_label[ds_index]
        new_data = torch.cat((new_data,anchor_data),0)
        new_label = torch.cat((new_label,anchor_label.cuda()),0)
        j = j+1

    ###############augment for minor cell types##################
    j = 0
    for anchor in minor_anchor_cts:
        aug_fold = int((anchor_fold[j]))    
        remaining_cell = int(train_median - (int(anchor_fold[j]))*np.array(train_minor[1])[j])
        index = (train_label == anchor).nonzero(as_tuple=True)[0]
        anchor_data = train_data[index.tolist(),:]
        anchor_label = train_label[index.tolist()]
        anchor_transfomr_dataset = MyDataset(anchor_data, anchor_label)
        anchor_dl = DataLoader(anchor_transfomr_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
        reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label, reconstructed_label),0)
        for i in range(aug_fold-1):
            reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
            reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
            reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = torch.cat((new_data,reconstructed_data),0)
            new_label = torch.cat((new_label,reconstructed_label.cuda()),0)

        reconstructed_data, reconstructed_label,real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        #add remaining cell
        N = range(np.array(train_minor[1])[j])
        ds_index = random.sample(N, remaining_cell)
        reconstructed_data = reconstructed_data[ds_index,:]
        reconstructed_label = reconstructed_label[ds_index]
        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label,reconstructed_label.cuda()),0)
        j = j+1               

#######load the model trained before augmentation#########
checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
if os.path.exists(checkpoint_tar):
    checkpoint = torch.load(checkpoint_tar)
    start_epoch = checkpoint['epoch']
    best_top1_acc = checkpoint['best_top1_acc']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("load successfully")

if args.save_simulated_result == True:
    if not os.path.exists('../result/simulation_result/{}'.format(mode)):
        os.makedirs('../result/simulation_result/{}'.format(mode))
        
    reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, train_dl)
    pd.DataFrame(reconstructed_data.cpu().numpy()).to_csv( '../result/simulation_result/{}/sim.csv'.format(mode))
    pd.DataFrame(real_data.cpu().numpy()).to_csv( '../result/simulation_result/{}/real.csv'.format(args.dataset))
    pd.DataFrame(reconstructed_label.cpu().numpy()).to_csv( '../result/simulation_result/{}/label.csv'.format(mode))

############process new data after augmentation###########
train_transformed_dataset = MyDataset(new_data, new_label)
train_dl = DataLoader(train_transformed_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)

############## train model ###########
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=int(args.epochs/2),classify_dim=args.classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)
checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
if os.path.exists(checkpoint_tar):
    checkpoint = torch.load(checkpoint_tar)
    start_epoch = checkpoint['epoch']
    best_top1_acc = checkpoint['best_top1_acc']
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    print("load successfully")
model,acc2,num1,train_num = train_model(model, train_dl, test_dl, lr=args.lr/10, epochs=int(args.epochs/2),classify_dim=args.classify_dim,best_top1_acc=0, save_path=model_save_path,feature_num=feature_num)
average2 = torch.mean(torch.Tensor(acc2))

if args.fs == True:
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("load successfully")
        
    classify_model = nn.Sequential(*list(model.children()))[0:2]
    deconv = IntegratedGradients(classify_model)

    train_data_fs = new_data
    train_label_fs = new_label

    for i in range(args.classify_dim):
        train_index_fs= torch.where(train_label_fs==i)
        train_index_fs = [t.cpu().numpy() for t in train_index_fs]
        train_index_fs = np.array(train_index_fs)
        train_data_each_celltype_fs = train_data_fs[train_index_fs,:].reshape(-1,feature_num)
    
        attribution = torch.zeros(1,feature_num)
        for j in range(train_data_each_celltype_fs.size(0)-1):
            attribution = attribution.cuda()+  torch.abs(deconv.attribute(train_data_each_celltype_fs[j:j+1,:], target=i))

        attribution_mean = torch.mean(attribution,dim=0)
        print("celltype:", i, "finish")

        if not os.path.exists(save_fs_eachcell):
            os.makedirs(save_fs_eachcell)
            
        pd.DataFrame(attribution_mean.cpu().numpy()).to_csv(save_fs_eachcell+"/fs."+".celltype"+str(i)+".csv")
    


if args.save_latent_space == True:
    #######build model#########
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("load successfully")

    simulated_data_ls, train_data_ls, label_ls = get_encodings(model,test_dl)
    simulated_data_ls[simulated_data_ls>torch.max(real_data)]=torch.max(train_data_ls)
    simulated_data_ls[simulated_data_ls<torch.min(real_data)]=torch.min(train_data_ls)
    simulated_data_ls[torch.isnan(simulated_data_ls)]=torch.max(train_data_ls)

    if not os.path.exists('../result/dimension_reduction/{}'.format(mode)):
        os.makedirs('../result/dimension_reduction/{}'.format(mode))
    pd.DataFrame(simulated_data_ls.cpu().numpy()).to_csv( '../result/dimension_reduction/{}/latent_space.csv'.format(mode))
    pd.DataFrame(label_ls.cpu().numpy()).to_csv('../result/dimension_reduction/{}/latent_space_label.csv'.format(mode))


output = []
output.append(float(np.array(average1)))
output.append(float(np.array(average2)))

output_v.append(list(output))
print("The average accuracy before/after augmentation are:", output_v)
    
            

