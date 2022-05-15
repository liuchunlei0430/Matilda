# Matilda: Multi-task learning from single cell multimodal omics

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from the multi-modality of such data and trains a neural network model to simultaneously learn multiple tasks including data simulation, dimension reduction, visualization, classification, and feature selection.

Matilda is developed using PyTorch 1.9.1 and requires >=1 GPU to run.

## Installation
Matilda can be obtained by simply clonning the github repository:

```
git clone https://github.com/liuchunlei0430/Matilda.git
```

The following python packages are required to be installed before running Matilda: h5py, torch, numpy, os, random, pandas, captum.

## Preparing intput for Matilda
### Example dataset 
As an example, the processed CITE-seq dataset by Ramaswamy et al. (GSE166489), SHARE-seq dataset by Ma et al. (GSE140203), TEA-seq dataset by Swanson et al. (GSE158013) from 10x Genomics can be downloaded from:

```
https://drive.google.com/drive/folders/1xdWzY0XLZkWYVD9XYTp_UALBhOKSmHAW?usp=sharing
```

Matilda’s main function takes expression data in .h5 format and cell type labels in .csv format. To prepare the example dataset as input for Matilda: 1) download the dataset from the above link and 2) modify dataset paths in data_processing_code/data_processing.Rmd.

## Running Matilda
In terminal, run

```
cd main

python main_matilda.py --nfeatures_rna 11062 --nfeatures_adt 189 --classify_dim 26 ###for CITE-seq

python main_matilda.py --nfeatures_rna 8926 --nfeatures_atac 14034 --classify_dim 22  ###for SHARE-seq

python main_matilda.py --nfeatures_rna 9855 --nfeatures_adt 46 --features_atac 14732 --classify_dim 11 ###for TEA-seq 
```

The output will be saved in ./output folder.

## Argument

### Dataset information

`nfeatures_rna`: Number of RNAs in the dataset.

`nfeatures_adt`: Number of ADTs in the dataset (can be null if atac is provided).

`nfeatures_atac`: Number of ATAC in the dataset (can be null if adt is provided). Note ATAC data should be summarised to the gene level as "gene activity score".

`classify_dim`: Number of cell types.

### Training and model config

`batch_size`: Batch size (set as 64 by default)

`epochs`: Number of epochs.

`lr`: Learning rate.

`z_dim`: Dimension of latent space.

`hidden_rna`: Dimension of RNA branch.

`hidden_adt`: Dimension of ADT branch.

`hidden_atac`: Dimension of ATAC branch.

### Other config

`seed`: The random seed for training.

`augmentation`: Whether to augment simulated data.

`fs`: Whether to perform feature selection.

`save_latent_space`: Whether to save the dimension reduction result.

`save_simulated_result`: Whether to save the simulation result.

`dataset`: Name of the input dataset.

### Example run
python main.py --nfeatures_rna 11062 --nfeatures_adt 189 --classify_dim 26

The dataset is CITEseq
 97%|██████████████████████████████████████████████████████████████████████████████████▏  | 29/30 [00:18<00:00,  1.62it/s]Epoch :  30 	
cell type :  0 	 	 prec : tensor(0., device='cuda:0') number: 59 train_cty_num: 15
cell type :  1 	 	 prec : tensor(89.8365, device='cuda:0') number: 830 train_cty_num: 207
cell type :  2 	 	 prec : tensor(84.5242, device='cuda:0') number: 912 train_cty_num: 228
cell type :  3 	 	 prec : tensor(22.2653, device='cuda:0') number: 548 train_cty_num: 136
cell type :  4 	 	 prec : tensor(10.0629, device='cuda:0') number: 70 train_cty_num: 18
cell type :  5 	 	 prec : tensor(97.4438, device='cuda:0') number: 571 train_cty_num: 143
cell type :  6 	 	 prec : tensor(94.9811, device='cuda:0') number: 972 train_cty_num: 243
cell type :  7 	 	 prec : tensor(45.2273, device='cuda:0') number: 287 train_cty_num: 72
cell type :  8 	 	 prec : tensor(99.2945, device='cuda:0') number: 688 train_cty_num: 172
cell type :  9 	 	 prec : tensor(92.6375, device='cuda:0') number: 626 train_cty_num: 157
cell type :  10 	 	 prec : tensor(85.6915, device='cuda:0') number: 188 train_cty_num: 47
cell type :  11 	 	 prec : tensor(81.5152, device='cuda:0') number: 81 train_cty_num: 20
cell type :  12 	 	 prec : tensor(3.5088, device='cuda:0') number: 47 train_cty_num: 12
cell type :  13 	 	 prec : tensor(10., device='cuda:0') number: 21 train_cty_num: 5
cell type :  14 	 	 prec : tensor(96.8605, device='cuda:0') number: 180 train_cty_num: 45
cell type :  15 	 	 prec : tensor(50.1026, device='cuda:0') number: 104 train_cty_num: 26
cell type :  16 	 	 prec : tensor(83.3333, device='cuda:0') number: 147 train_cty_num: 36
cell type :  17 	 	 prec : tensor(100., device='cuda:0') number: 12 train_cty_num: 4
cell type :  18 	 	 prec : tensor(43.1481, device='cuda:0') number: 61 train_cty_num: 15
cell type :  19 	 	 prec : tensor(47.5000, device='cuda:0') number: 22 train_cty_num: 6
cell type :  20 	 	 prec : tensor(41.3043, device='cuda:0') number: 26 train_cty_num: 6
cell type :  21 	 	 prec : tensor(29.8507, device='cuda:0') number: 94 train_cty_num: 23
cell type :  22 	 	 prec : tensor(63.5135, device='cuda:0') number: 55 train_cty_num: 14
cell type :  23 	 	 prec : tensor(96.6667, device='cuda:0') number: 65 train_cty_num: 16
cell type :  24 	 	 prec : tensor(96.3235, device='cuda:0') number: 229 train_cty_num: 57
cell type :  25 	 	 prec : tensor(94.1176, device='cuda:0') number: 18 train_cty_num: 5
100%|█████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:19<00:00,  1.55it/s]
load successfully
 93%|███████████████████████████████████████████████████████████████████████████████▎     | 14/15 [00:08<00:00,  1.77it/s]Epoch :  15 	
cell type :  0 	 	 prec : tensor(23.8889, device='cuda:0') number: 59 train_cty_num: 23
cell type :  1 	 	 prec : tensor(84.8866, device='cuda:0') number: 830 train_cty_num: 23
cell type :  2 	 	 prec : tensor(89.8338, device='cuda:0') number: 912 train_cty_num: 23
cell type :  3 	 	 prec : tensor(1.3804, device='cuda:0') number: 548 train_cty_num: 23
cell type :  4 	 	 prec : tensor(66.6667, device='cuda:0') number: 70 train_cty_num: 23
cell type :  5 	 	 prec : tensor(87.8185, device='cuda:0') number: 571 train_cty_num: 23
cell type :  6 	 	 prec : tensor(92.7741, device='cuda:0') number: 972 train_cty_num: 23
cell type :  7 	 	 prec : tensor(54.6163, device='cuda:0') number: 287 train_cty_num: 23
cell type :  8 	 	 prec : tensor(99.4727, device='cuda:0') number: 688 train_cty_num: 23
cell type :  9 	 	 prec : tensor(71.1498, device='cuda:0') number: 626 train_cty_num: 23
cell type :  10 	 	 prec : tensor(93.2407, device='cuda:0') number: 188 train_cty_num: 23
cell type :  11 	 	 prec : tensor(88.0952, device='cuda:0') number: 81 train_cty_num: 23
cell type :  12 	 	 prec : tensor(52.7027, device='cuda:0') number: 47 train_cty_num: 23
cell type :  13 	 	 prec : tensor(80., device='cuda:0') number: 21 train_cty_num: 23
cell type :  14 	 	 prec : tensor(93.1648, device='cuda:0') number: 180 train_cty_num: 23
cell type :  15 	 	 prec : tensor(65.1961, device='cuda:0') number: 104 train_cty_num: 23
cell type :  16 	 	 prec : tensor(91.4557, device='cuda:0') number: 147 train_cty_num: 23
cell type :  17 	 	 prec : tensor(100., device='cuda:0') number: 12 train_cty_num: 23
cell type :  18 	 	 prec : tensor(22.3404, device='cuda:0') number: 61 train_cty_num: 23
cell type :  19 	 	 prec : tensor(66.6667, device='cuda:0') number: 22 train_cty_num: 23
cell type :  20 	 	 prec : tensor(69.5652, device='cuda:0') number: 26 train_cty_num: 23
cell type :  21 	 	 prec : tensor(53.3898, device='cuda:0') number: 94 train_cty_num: 23
cell type :  22 	 	 prec : tensor(88.2979, device='cuda:0') number: 55 train_cty_num: 23
cell type :  23 	 	 prec : tensor(97., device='cuda:0') number: 65 train_cty_num: 23
cell type :  24 	 	 prec : tensor(97.3611, device='cuda:0') number: 229 train_cty_num: 23
cell type :  25 	 	 prec : tensor(88.8889, device='cuda:0') number: 18 train_cty_num: 23
100%|█████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:08<00:00,  1.72it/s]
load successfully
 93%|███████████████████████████████████████████████████████████████████████████████▎     | 14/15 [00:07<00:00,  1.82it/s]Epoch :  15 	
cell type :  0 	 	 prec : tensor(46.0993, device='cuda:0') number: 59 train_cty_num: 23
cell type :  1 	 	 prec : tensor(83.1609, device='cuda:0') number: 830 train_cty_num: 23
cell type :  2 	 	 prec : tensor(89.9341, device='cuda:0') number: 912 train_cty_num: 23
cell type :  3 	 	 prec : tensor(9.9596, device='cuda:0') number: 548 train_cty_num: 23
cell type :  4 	 	 prec : tensor(78.2313, device='cuda:0') number: 70 train_cty_num: 23
cell type :  5 	 	 prec : tensor(88.1613, device='cuda:0') number: 571 train_cty_num: 23
cell type :  6 	 	 prec : tensor(93.2085, device='cuda:0') number: 972 train_cty_num: 23
cell type :  7 	 	 prec : tensor(55.7048, device='cuda:0') number: 287 train_cty_num: 23
cell type :  8 	 	 prec : tensor(98.8405, device='cuda:0') number: 688 train_cty_num: 23
cell type :  9 	 	 prec : tensor(81.4317, device='cuda:0') number: 626 train_cty_num: 23
cell type :  10 	 	 prec : tensor(96.6667, device='cuda:0') number: 188 train_cty_num: 23
cell type :  11 	 	 prec : tensor(89.5115, device='cuda:0') number: 81 train_cty_num: 23
cell type :  12 	 	 prec : tensor(65.4167, device='cuda:0') number: 47 train_cty_num: 23
cell type :  13 	 	 prec : tensor(83.3333, device='cuda:0') number: 21 train_cty_num: 23
cell type :  14 	 	 prec : tensor(93.5185, device='cuda:0') number: 180 train_cty_num: 23
cell type :  15 	 	 prec : tensor(66.4516, device='cuda:0') number: 104 train_cty_num: 23
cell type :  16 	 	 prec : tensor(93.6508, device='cuda:0') number: 147 train_cty_num: 23
cell type :  17 	 	 prec : tensor(100., device='cuda:0') number: 12 train_cty_num: 23
cell type :  18 	 	 prec : tensor(66.3120, device='cuda:0') number: 61 train_cty_num: 23
cell type :  19 	 	 prec : tensor(47.2222, device='cuda:0') number: 22 train_cty_num: 23
cell type :  20 	 	 prec : tensor(71.7391, device='cuda:0') number: 26 train_cty_num: 23
cell type :  21 	 	 prec : tensor(43.5714, device='cuda:0') number: 94 train_cty_num: 23
cell type :  22 	 	 prec : tensor(90.6504, device='cuda:0') number: 55 train_cty_num: 23
cell type :  23 	 	 prec : tensor(94.5578, device='cuda:0') number: 65 train_cty_num: 23
cell type :  24 	 	 prec : tensor(94.6057, device='cuda:0') number: 229 train_cty_num: 23
cell type :  25 	 	 prec : tensor(100., device='cuda:0') number: 18 train_cty_num: 23
100%|█████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:08<00:00,  1.76it/s]
load successfully
celltype: 0 finish
celltype: 1 finish
celltype: 2 finish
celltype: 3 finish
celltype: 4 finish
celltype: 5 finish
celltype: 6 finish
celltype: 7 finish
celltype: 8 finish
celltype: 9 finish
celltype: 10 finish
celltype: 11 finish
celltype: 12 finish
celltype: 13 finish
celltype: 14 finish
celltype: 15 finish
celltype: 16 finish
celltype: 17 finish
celltype: 18 finish
celltype: 19 finish
celltype: 20 finish
celltype: 21 finish
celltype: 22 finish
celltype: 23 finish
celltype: 24 finish
celltype: 25 finish
load successfully
The average accuracy before/after augmentation are: [[63.834983825683594, 77.76691436767578]]


## Output

After training, Matilda will output the average accuracy before augmentation and after augmentation.

## Reference
Stephenson, E. et al. Single-cell multi-omics analysis of the immune response in COVID-19. Nat. Med. 27, 904–916 (2021).
