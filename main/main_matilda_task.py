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
from learn.predict import test_model
from util import setup_seed, real_label, MyDataset,ToTensor, read_h5_data, read_fs_label, get_vae_simulated_data_from_sampling, get_encodings, compute_zscore, compute_log2

parser = argparse.ArgumentParser("Matilda")
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--classification', type=bool, default= False, help='if augmentation or not')
parser.add_argument('--query', type=bool, default= False, help='if the data is query of reference')
parser.add_argument('--fs', type=bool, default= False, help='if doing feature selection or not')
parser.add_argument('--dim_reduce', type=bool, default= False, help='save latent space')
parser.add_argument('--simulation', type=bool, default= False, help='save simulation result')
parser.add_argument('--simulation_ct', type=int, default= 1, help='save simulation result')
parser.add_argument('--simulation_num', type=int, default= 100, help='save simulation result')

############# for data build ##############
parser.add_argument('--rna', metavar='DIR', default='NULL', help='path to train rna data')
parser.add_argument('--adt', metavar='DIR', default='NULL', help='path to train adt data')
parser.add_argument('--atac', metavar='DIR', default='NULL', help='path to train atac data')
parser.add_argument('--cty', metavar='DIR', default='NULL', help='path to train cell type label')

##############  for training #################
parser.add_argument('--batch_size', type=int, default=64, help='batch size')


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

if args.query:
    path = "query"
else:
    path = "reference"
    
if args.adt != "NULL" and args.atac != "NULL":
    mode = "TEAseq"
    rna_data_path = args.rna
    adt_data_path = args.adt
    atac_data_path = args.atac
    label_path = args.cty
    rna_data = read_h5_data(rna_data_path)
    adt_data = read_h5_data(adt_data_path)
    atac_data = read_h5_data(atac_data_path)
    label = read_fs_label(label_path)
    classify_dim = (max(label)+1).cpu().numpy()
    nfeatures_rna = rna_data.shape[1]
    nfeatures_adt = adt_data.shape[1]
    nfeatures_atac = atac_data.shape[1]
    feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac
    rna_data = compute_log2(rna_data)
    adt_data = compute_log2(adt_data)
    atac_data = compute_log2(atac_data)
    rna_data = compute_zscore(rna_data)
    adt_data = compute_zscore(adt_data)
    atac_data = compute_zscore(atac_data)
    data = torch.cat((rna_data,adt_data,atac_data),1)       
    transformed_dataset = MyDataset(data, label)
    dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

        
print("The dataset is", mode)    
output_v = []
model_save_path = "../trained_model/{}/".format(mode)    
save_fs_eachcell = "../output/marker/{}/{}/".format(mode,path)   

transform_real_label = real_label(label_path, classify_dim)
#######build model#########
if mode == "CITEseq":
    model = CiteAutoencoder_CITEseq(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
elif mode == "SHAREseq":
    model = CiteAutoencoder_SHAREseq(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
elif mode == "TEAseq":
    model = CiteAutoencoder_TEAseq(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)

#model = nn.DataParallel(model).cuda() #multi gpu
model = model.cuda() #one gpu
########train model#########

if args.classification == True:  
    if not os.path.exists('../output/classification/{}/{}'.format(mode,path)):
        os.makedirs('../output/classification/{}/{}'.format(mode,path))
    save_path = open('../output/classification/{}/{}/accuracy_each_ct.txt'.format(mode,path),"w")
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    model, acc1, num1 = test_model(model, dl, transform_real_label, classify_dim = classify_dim, save_path = save_path)
    average1 = torch.mean(torch.Tensor(acc1))



if args.simulation == True:
    print("simulate celltype",  args.simulation_ct)
    checkpoint_tar = os.path.join(model_save_path, 'simulation_model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    if not os.path.exists('../output/simulation_result/{}/{}/'.format(mode,path)):
        os.makedirs('../output/simulation_result/{}/{}/'.format(mode,path))        
        
    index = (label == args.simulation_ct).nonzero(as_tuple=True)[0]
    aug_fold = int(args.simulation_num/int(index.size(0)))    
    remaining_cell = int(args.simulation_num - aug_fold*int(index.size(0)))

    index = (label == args.simulation_ct).nonzero(as_tuple=True)[0]
    anchor_data = data[index.tolist(),:]
    anchor_label = label[index.tolist()]
    anchor_transform_dataset = MyDataset(anchor_data, anchor_label)
    anchor_dl = DataLoader(anchor_transform_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
    new_data = []
    new_label = []
    
    if aug_fold >= 1:
        reconstructed_data, reconstructed_label, real_data = get_vae_simulated_data_from_sampling(model, anchor_dl)
        reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data)
        reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data)
        reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)

        new_data = reconstructed_data 
        new_label = reconstructed_label
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
    N = range(np.array(reconstructed_data.size(0)))
    ds_index = random.sample(N, remaining_cell)
    reconstructed_data = reconstructed_data[ds_index,:]
    reconstructed_label = reconstructed_label[ds_index]
    if aug_fold ==0:
        new_data = reconstructed_data
        new_label = reconstructed_label
    else:
        new_data = torch.cat((new_data,reconstructed_data),0)
        new_label = torch.cat((new_label,reconstructed_label.cuda()),0)

    index = (label != args.simulation_ct).nonzero(as_tuple=True)[0]
    anchor_data = data[index.tolist(),:]
    anchor_label = label[index.tolist()]
    real_data = data
    real_label = label
    sim_data = torch.cat((anchor_data,new_data),0)
    sim_label = torch.cat((anchor_label,new_label.cuda()),0)

    pd.DataFrame(sim_data.cpu().numpy()).to_csv( '../output/simulation_result/{}/{}/sim_data.csv'.format(mode,path))
    pd.DataFrame(real_data.cpu().numpy()).to_csv( '../output/simulation_result/{}/{}/real_data.csv'.format(mode,path))
    pd.DataFrame(sim_label.cpu().numpy()).to_csv( '../output/simulation_result/{}/{}/sim_label.csv'.format(mode,path))
    pd.DataFrame(real_label.cpu().numpy()).to_csv( '../output/simulation_result/{}/{}/real_label.csv'.format(mode,path))
    
    print("finish simulation")
    
if args.dim_reduce == True:
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    simulated_data_ls, data_ls, label_ls = get_encodings(model,dl)
    simulated_data_ls[simulated_data_ls>torch.max(data)]=torch.max(data_ls)
    simulated_data_ls[simulated_data_ls<torch.min(data)]=torch.min(data_ls)
    simulated_data_ls[torch.isnan(simulated_data_ls)]=torch.max(data_ls)
    if not os.path.exists('../output/visualisation/{}/{}/'.format(mode,path)):
        os.makedirs('../output/visualisation/{}/{}/'.format(mode,path))
    pd.DataFrame(simulated_data_ls.cpu().numpy()).to_csv( '../output/visualisation/{}/{}/latent_space.csv'.format(mode,path))
    pd.DataFrame(label_ls.cpu().numpy()).to_csv('../output/visualisation/{}/{}/latent_space_label.csv'.format(mode,path))
    print("finish dimension reduction")

if args.fs == True:
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
    classify_model = nn.Sequential(*list(model.children()))[0:2]
    deconv = IntegratedGradients(classify_model)
    for i in range(classify_dim):
        train_index_fs= torch.where(label==i)
        train_index_fs = [t.cpu().numpy() for t in train_index_fs]
        train_index_fs = np.array(train_index_fs)
        train_data_each_celltype_fs = data[train_index_fs,:].reshape(-1,feature_num)
    
        attribution = torch.zeros(1,feature_num)
        for j in range(train_data_each_celltype_fs.size(0)-1):
            attribution = attribution.cuda()+  torch.abs(deconv.attribute(train_data_each_celltype_fs[j:j+1,:], target=i))
        attribution_mean = torch.mean(attribution,dim=0)

        if not os.path.exists(save_fs_eachcell):
            os.makedirs(save_fs_eachcell)   
        pd.DataFrame(attribution_mean.cpu().numpy()).to_csv(save_fs_eachcell+"/fs."+"celltype"+str(i)+".csv")
    print("finish feature selection")
    
            

