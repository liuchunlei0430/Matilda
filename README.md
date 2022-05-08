# Matilda: Multi-task learning from single cell multimodal omics with Matilda

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from multimodal data and uses a neural network to train the model, enabling the tasks including data simulation, dimension reduction, visualization, classification, and feature selection. For more information, please see Matilda manuscript.
Matilda is developed using PyTorch 1.9.1 and requires 1 GPU to run.

## Installation
Matilda can be obtained by simply clonning the github repository:

```
git clone https://github.com/liuchunlei0430/Matilda.git
```

The following python packages are required to be installed before running Matilda: h5py, torch, numpy, os, random, pandas, captum.

## Preparing intput for Matilda
### Dataset
The processed data from 10x Genomics and EMBL-EBI ArrayExpress database can be downloaded from:

```
https://drive.google.com/drive/folders/1uuSIiURzAUtu7r9V2wHwXbs7S_1srGg9?usp=sharing
```

Matilda’s main function takes expression data in .h5 format and cell type labels in .csv format. To prepare the input for Matilda: 1) download the datasets 2) modifying dataset paths in data_processing_code/data_processing.Rmd.

## Running Matilda
In terminal, run

```
cd main
python main_citeseq.py
```

The output will be saved in ./output folder.

## Argument

### Dataset information

`feature_num`: Number of total features in both training and test data for multi-omic data.

`nfeatures_rna`: Number of RNA expressions in both training and test data for multi-omic data.

`nfeatures_pro`: Number of ADT expressions in both training and test data for multi-omic data. Note in 2-modality data, it may 
represents the number of ATAC expressions.

`classify_dim`: Number of cell type.

### Training and model config

`batch_size`: Batch size (set as 64 by default)

`epochs`: Number of epochs.

`lr`: Learning rate.

`z_dim`: Dimension of latent space.

`hidden_rna`: Dimension of RNA branch.

`hidden_pro`: Dimension of ADT branch.

`hidden_atac`: Dimension of ATAC branch.

### Other config

`seed`: The random seed set in training.

`augmentation`: Whether do augmentation or not.

`fs`: Whether do feature selection or not.

`save_latent_space`: Whether save the dimension reduction result or not.

`save_simulated_result`: Whether save the simulation result or not.

`dataset`: Name of the dataset.


## Output

After training, Matilda will output the average accuracy before augmentation and after augmentation.
