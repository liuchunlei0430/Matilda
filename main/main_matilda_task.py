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
import h5py,scipy

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
    dl = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        
print("The dataset is", mode)    
output_v = []
model_save_path = "../trained_model/{}/".format(mode)    
save_fs_eachcell = "../output/marker/{}/{}/".format(mode,path)   


output_v = []

rna_name  = h5py.File(rna_data_path,"r")['matrix/features'][:]
adt_name  = h5py.File(adt_data_path,"r")['matrix/features'][:]
atac_name  = h5py.File(atac_data_path,"r")['matrix/features'][:]

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
    save_path1 = open('../output/classification/{}/{}/accuracy_each_cell.txt'.format(mode,path),"w")
    checkpoint_tar = os.path.join(model_save_path, 'model_best.pth.tar')
    if os.path.exists(checkpoint_tar):
        checkpoint = torch.load(checkpoint_tar)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    model, acc1, num1,classified_label, groundtruth_label,prob = test_model(model, dl, transform_real_label, classify_dim = classify_dim, save_path = save_path)
    average1 = torch.mean(torch.Tensor(acc1))
    for j in range(len(groundtruth_label)):
        print('cell ID: ',j, '\t', '\t', 'real cell type:', groundtruth_label[j], '\t', '\t', 'predicted cell type:', classified_label[j], '\t', '\t', 'probability:', round(float(prob[j]),2), file = save_path1)
    


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
    sim_data_rna = sim_data[:, 0:nfeatures_rna]
    real_data_rna = real_data[:, 0:nfeatures_rna]   
    if mode == "CITEseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_adt = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
    elif mode == "SHAREseq":
        sim_data_atac = sim_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        real_data_atac = real_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
    elif mode == "TEAseq":
        sim_data_adt = sim_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        sim_data_atac = sim_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]
        real_data_adt = real_data[:, nfeatures_rna:(nfeatures_rna+nfeatures_adt)]
        real_data_atac = real_data[:, (nfeatures_rna+nfeatures_adt):(nfeatures_rna+nfeatures_adt+nfeatures_atac)]

    rna_name_new = []
    adt_name_new = []
    atac_name_new = []
    
    b_list = range(0, real_data_rna.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]
    b_list = range(0, sim_data_rna.size(0))
    cell_name_sim = ['cell_{}'.format(b) for b in b_list]
    sim_label_new = []
    real_label_new = []
    for j in range(sim_data_rna.size(0)):
        sim_label_new.append(transform_real_label[sim_label[j]])
    for j in range(real_data_rna.size(0)):    
        real_label_new.append(transform_real_label[real_label[j]])
    for i in range(sim_data_rna.size(1)):
        a = str(rna_name[i],encoding="utf-8")
        rna_name_new.append(a)           
        
    if mode == "CITEseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        pd.DataFrame(sim_data_adt.cpu().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))

        
    if mode == "SHAREseq":
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)           
        pd.DataFrame(sim_data_atac.cpu().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

        
    if mode == "TEAseq":
        for i in range(sim_data_adt.size(1)):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        for i in range(sim_data_atac.size(1)):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)            
        pd.DataFrame(sim_data_adt.cpu().numpy(), index = cell_name_sim, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_adt.csv'.format(mode,path))
        pd.DataFrame(real_data_adt.cpu().numpy(), index = cell_name_real, columns = adt_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_adt.csv'.format(mode,path))
        pd.DataFrame(sim_data_atac.cpu().numpy(), index = cell_name_sim, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_atac.csv'.format(mode,path))
        pd.DataFrame(real_data_atac.cpu().numpy(), index = cell_name_real, columns = atac_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_atac.csv'.format(mode,path))

    pd.DataFrame(sim_data_rna.cpu().numpy(), index = cell_name_sim, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/sim_data_rna.csv'.format(mode,path))
    pd.DataFrame(real_data_rna.cpu().numpy(), index = cell_name_real, columns = rna_name_new).to_csv( '../output/simulation_result/{}/{}/real_data_rna.csv'.format(mode,path))
    pd.DataFrame(sim_label_new,  index = cell_name_sim, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/sim_label.csv'.format(mode,path))
    pd.DataFrame(real_label_new,  index = cell_name_real, columns = [ "label"]).to_csv( '../output/simulation_result/{}/{}/real_label.csv'.format(mode,path))

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
    if not os.path.exists('../output/dim_reduce/{}/{}/'.format(mode,path)):
        os.makedirs('../output/dim_reduce/{}/{}/'.format(mode,path))
    b_list = range(0, simulated_data_ls.size(1))
    feature_index = ['feature_{}'.format(b) for b in b_list]   
    b_list = range(0, data.size(0))
    cell_name_real = ['cell_{}'.format(b) for b in b_list]  
    
    real_label_new = []
    for j in range(data.size(0)):    
        real_label_new.append(transform_real_label[label[j]])
            
    pd.DataFrame(simulated_data_ls.cpu().numpy(), index = cell_name_real, columns = feature_index).to_csv( '../output/dim_reduce/{}/{}/latent_space.csv'.format(mode,path))
    pd.DataFrame(real_label_new, index = cell_name_real, columns = [ "label"]).to_csv('../output/dim_reduce/{}/{}/latent_space_label.csv'.format(mode,path))
    print("finish dimension reduction")

if args.fs == True:
    rna_name_new = []
    adt_name_new = []
    atac_name_new = []
    for i in range(nfeatures_rna):
        a = str(rna_name[i],encoding="utf-8")
        rna_name_new.append(a)
    if mode == "CITEseq":
        for i in range(nfeatures_adt):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        features = rna_name_new + adt_name_new
    if mode == "SHAREseq":
        for i in range(nfeatures_atac):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)
        features = rna_name_new + atac_name_new
    if mode == "TEAseq":
        for i in range(nfeatures_adt):
            a = str(adt_name[i],encoding="utf-8")
            adt_name_new.append(a)
        for i in range(nfeatures_atac):
            a = str(atac_name[i],encoding="utf-8")
            atac_name_new.append(a)
        features = rna_name_new + adt_name_new + atac_name_new
        
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
        pd.DataFrame(attribution_mean.cpu().numpy(), index = features, columns = [ "importance score"]).to_csv(save_fs_eachcell+"/fs."+"celltype"+str(i)+".csv")
    print("finish feature selection")
    

