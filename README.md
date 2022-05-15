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
### Example dataset (Stephenson et al.)
As an example, the processed CITE-seq dataset by Stephenson et al. from 10x Genomics and EMBL-EBI ArrayExpress database can be downloaded from:

```
https://drive.google.com/drive/folders/1uuSIiURzAUtu7r9V2wHwXbs7S_1srGg9?usp=sharing
```

Matilda’s main function takes expression data in .h5 format and cell type labels in .csv format. To prepare the example dataset as input for Matilda: 1) download the dataset from the above link and 2) modify dataset paths in data_processing_code/data_processing.Rmd.

## Running Matilda
In terminal, run

```
cd main
python main_matilda.py
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

python main.py --nfeatures_rna 8926 --nfeatures_atac 14034 --classify_dim 22

python main.py --nfeatures_rna 9855 --nfeatures_adt 46 --features_atac 14732 --classify_dim 11


## Output

After training, Matilda will output the average accuracy before augmentation and after augmentation.

## Reference
Stephenson, E. et al. Single-cell multi-omics analysis of the immune response in COVID-19. Nat. Med. 27, 904–916 (2021).
