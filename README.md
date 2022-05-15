# Matilda: Multi-task learning from single-cell multimodal omics

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from the multi-modality of such data and trains a neural network model to simultaneously learn multiple tasks including data simulation, dimension reduction, visualization, classification, and feature selection.

<img width=100% src="https://github.com/liuchunlei0430/Matilda/blob/main/img/main.jpg"/>

Matilda is developed using PyTorch 1.9.1 and requires >=1 GPU to run.

## Installation
Matilda can be obtained by simply clonning the github repository:

```
git clone https://github.com/liuchunlei0430/Matilda.git
```

The following python packages are required to be installed before running Matilda: h5py, torch, numpy, os, random, pandas, captum.

## Preparing intput for Matilda
### Example dataset 
As an example, the processed CITE-seq dataset by Ramaswamy et al. (GSE166489), SHARE-seq dataset by Ma et al. (GSE140203), TEA-seq dataset by Swanson et al. (GSE158013) from 10x Genomics can be downloaded from [link](https://drive.google.com/drive/folders/1xdWzY0XLZkWYVD9XYTp_UALBhOKSmHAW?usp=sharing):

```
https://drive.google.com/drive/folders/1xdWzY0XLZkWYVD9XYTp_UALBhOKSmHAW?usp=sharing
```

Matilda’s main function takes expression data in .h5 format and cell type labels in .csv format. To prepare the example dataset as input for Matilda: 1) download the dataset from the above link and 2) modify dataset paths in data_processing_code/data_processing.Rmd.

## Running Matilda

In terminal, setting arguments according to the data input (See Arguments section for more details).

```
cd main

python main_matilda.py --nfeatures_rna 11062 --nfeatures_adt 189 --classify_dim 26 ###for CITE-seq

python main_matilda.py --nfeatures_rna 8926 --nfeatures_atac 14034 --classify_dim 22  ###for SHARE-seq

python main_matilda.py --nfeatures_rna 9855 --nfeatures_adt 46 --features_atac 14732 --classify_dim 11 ###for TEA-seq 
```

The output will be saved in ./output folder.

## Argument

### Dataset information

+ `nfeatures_rna`: Number of RNAs in the dataset.

+ `nfeatures_adt`: Number of ADTs in the dataset (can be null if atac is provided).

+ `nfeatures_atac`: Number of ATAC in the dataset (can be null if adt is provided). Note ATAC data should be summarised to the gene level as "gene activity score".

+ `classify_dim`: Number of cell types.

### Training and model config

+ `batch_size`: Batch size (set as 64 by default)

+ `epochs`: Number of epochs.

+ `lr`: Learning rate.

+ `z_dim`: Dimension of latent space.

+ `hidden_rna`: Dimension of RNA branch.

+ `hidden_adt`: Dimension of ADT branch.

+ `hidden_atac`: Dimension of ATAC branch.

### Other config

+ `seed`: The random seed for training.

+ `augmentation`: Whether to augment simulated data.

+ `fs`: Whether to perform feature selection.

+ `save_latent_space`: Whether to save the dimension reduction result.

+ `save_simulated_result`: Whether to save the simulation result.

### Example run
python main_matilda.py --nfeatures_rna 11062 --nfeatures_adt 189 --classify_dim 26

## Output

After training, Matilda has 4 types of outputs:

1) the average accuracy before augmentation and after augmentation saved in `./output/classification/`; 

2) the simulated data saved in `./output/simulation_result/`; 

3) the latent space for dimension reduction saved in `./output/dimension_reduction/`; 
 
4) the joint markers saved in `./output/marker/`.

## Visualisation
To generate UMAP plots for the simulated data using R, modify dataset paths in './qc/visualize_simulated_data.Rmd' and run the following codes in terminal:

```
cd qc

Rscript visualize_simulated_data.Rmd
```

To generate UMAP plots and ARI, NMI, FM, Jaccard for the latent space using R, modify dataset paths in './qc/visualize_latent_space.Rmd' and run the following codes in terminal:

```
cd qc

Rscript visualize_latent_space.Rmd
```

## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).

[2] Ma, A., McDermaid, A., Xu, J., Chang, Y. & Ma, Q. Integrative Methods and Practical Challenges
for Single-Cell Multi-omics. Trends Biotechnol. 38, 1007–1022 (2020).

[3] Swanson, E. et al. Simultaneous trimodal single-cell measurement of transcripts, epitopes, and
chromatin accessibility using TEA-seq. Elife 10, (2021).
