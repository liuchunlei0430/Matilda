# Matilda: Multi-task learning from single-cell multimodal omics

Matilda is a multi-task framework for learning from single-cell multimodal omics data. Matilda leverages the information from the multi-modality of such data and trains a neural network model to simultaneously learn multiple tasks including data simulation, dimension reduction, visualization, classification, and feature selection.

<img width=100% src="https://github.com/liuchunlei0430/Matilda/blob/main/img/main.jpg"/>

Matilda is developed using PyTorch 1.9.1 and requires >=1 GPU to run.

## Installation
We use Ubuntu system. We recommend to use conda enviroment to install and run Matilda. We assume conda is installed.

Step 1:
Create and activate the conda environment for matilda
```
conda create -n environment_matilda python=3.8
conda activate environment_matilda
```

Step 2:
Check the environment including GPU settings and the highest CUDA version allowed by the GPU.
```
nvidia-smi
```

Step 3:
Install pytorch and cuda version based on your GPU settings.
```
# Example code for installing CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Step 4:
The following python packages are required to be installed before running Matilda: h5py, torch, numpy, os, random, pandas, captum. We will install these packages in the conda environment as the following:
```
pip install h5py
pip install numpy
pip install pandas
pip install captum
```

Step 5:
Otain Matilda by clonning the github repository:
```
git clone https://github.com/liuchunlei0430/Matilda.git
```

## Preparing intput for Matilda
Matilda’s main function takes expression data (e.g., RNA, ADT, ATAC) in `.h5` format and cell type labels in `.csv` format. Matilda expects raw count data for RNA and ADT modalities. For ATAC modality, Matilda expects the 'gene activity score' generated by Seurat from raw count data.

An example for creating gene activity score from ATAC modality in the R environment using human gene annotation is as below:
```
gene.activities <- CreateGeneActivityMatrix2(peak.matrix=teaseq.peak,
                                             annotation.file = “Homo_sapiens.GRCh38.90.chr.gtf.gz”,
                                             seq.levels = c(1:22, “X”, “Y”),
                                             seq_replace = c(“:”))
```

### Example dataset
First,  we create the `./data` folder under `Matilda`:
```
cd Matilda
mkdir data
```

As an example, the processed TEA-seq dataset by Swanson et al. (GSE158013) is provided for the example run, which can be downloaded from [link](https://drive.google.com/file/d/1ojilvNBB95GbgtF9a0IHl3yWQwl6gDl8/view?usp=sharing) or the below command:
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ojilvNBB95GbgtF9a0IHl3yWQwl6gDl8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ojilvNBB95GbgtF9a0IHl3yWQwl6gDl8" -O TEAseq.zip && rm -rf /tmp/cookies.txt
unzip TEAseq.zip
```

Users can prepare the example dataset as input for Matilda by downloading the dataset from the above link or use their own datasets.

## Running Matilda with the example dataset
### Training the Matilda model (see Arguments section for more details).
```
cd main
# training the matilda model
python main_matilda_train.py --rna [trainRNA] --adt [trainADT] --atac [trainATAC] #[training dataset]
```

### Argument

Training dataset information
+ `--rna`: path to training data RNA modality.
+ `--adt`: path to training data ADT modality (can be null if ATAC is provided).
+ `--atac`: path to training data ATAC modality (can be null if ADT is provided). Note ATAC data should be summarised to the gene level as "gene activity score".

Training and model config
+ `batch_size`: Batch size (set as 64 by default)
+ `epochs`: Number of epochs.
+ `lr`: Learning rate.
+ `z_dim`: Dimension of latent space.
+ `hidden_rna`: Dimension of RNA branch.
+ `hidden_adt`: Dimension of ADT branch.
+ `hidden_atac`: Dimension of ATAC branch.

Other config
+ `seed`: The random seed for training.
+ `augmentation`: Whether to augment simulated data.


### Perform multiple tasks using trained Matilda model.
Multi-task on the training data
```
# using the trained model for data simulation
python main_matilda_task.py --simulate [cellType] -n 200
```

output

```
# using the trained model for data visualisation
python main_matilda_task.py --visualisation
```

output

```
# using the trained model for feature selection
python main_matilda_task.py --fs
```

output


Multi-task using query data
```
# using the trained model for classifying query data
python main_matilda_task.py --classify --rna [queryRNA] --adt [queryADT] --atac [queryATAC]
```

output

```
# using the trained model for visualising query data
python main_matilda_task.py --visualisation --rna [queryRNA] --adt [queryADT] --atac [queryATAC]
```

output

```
# using the trained model for feature selection
python main_matilda_task.py --fs --rna [queryRNA] --adt [queryADT] --atac [queryATAC]
```

output

The output will be saved in ./output folder.


## Output

After training, Matilda has 4 types of outputs:
1) Cell type classification: the average accuracy before augmentation and after augmentation saved in `./output/classification/`;
2) Simulated data: simulated dataset saved in `./output/simulation_result/`; 
3) the latent space for dimension reduction saved in `./output/dimension_reduction/`; 
4) the joint markers saved in `./output/marker/`.

## Visualisation
To generate UMAP plots for the simulated data using R, modify dataset paths in `./qc/visualize_simulated_data.Rmd` and run this file.

To generate UMAP plots and ARI, NMI, FM, Jaccard for the latent space using R, modify dataset paths in `./qc/visualize_latent_space.Rmd` and run this file.

## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).

[2] Ma, A., McDermaid, A., Xu, J., Chang, Y. & Ma, Q. Integrative Methods and Practical Challenges
for Single-Cell Multi-omics. Trends Biotechnol. 38, 1007–1022 (2020).

[3] Swanson, E. et al. Simultaneous trimodal single-cell measurement of transcripts, epitopes, and
chromatin accessibility using TEA-seq. Elife 10, (2021).
