# Industrial anomaly detection task : PatchCore algorithm

## Presentation 
The goal is to build a model that is able to automatically detect anomalies (from subtle detail changes to large structural defects) in industrial products despite having been trained on normal samples only. It will focus on the industrial setting and in particular the “cold-start” problem which refers to the difficulty of identifying abnormal patterns or behaviors in a new dataset or system that has not been previously seen or used. 

To deploy this project, we use the provided paper: [“Towards Total Recall in Industrial Anomaly Detection”](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf) which proposes a new approach for the cold-start anomaly detection problem called **“PatchCore”**. 

PatchCore is an approach that has some specificities:
- The usage of **mid-level features** in order to extract for each image a set of features each one representing a specific image patch because using deep level features may cause harm if the layer’s choice is not carefully considered since they become more and more biased towards the task on which the model has been trained.
- The implementation of a **coreset-reduced patch-feature memory bank** in order to reduce cost and time of computation execution by selecting patch-features that maximizes the coverage of the original set.

For this project, we tried to re-implement and understand all the speficities of the PatchCore algorithm appraoch, using the [MV Tect Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad/) dataset to train and test it, a specific dataset designed to benchmark industrial anomaly detection approaches. 

[Official implementation code](github.com/amazon-research/patchcore-inspection)   
[Easier implementation (on which we based our implementation) code](https://github.com/rvorias/ind_knn_ad)

![image](https://user-images.githubusercontent.com/76529865/214596320-fb91598d-4fe9-4d2d-886b-49216d621fef.png)

## Table of content 
1. [MVTec image folder](#C_mvtec_anomaly_detection/bottle) : The folder that will contains the images once loaded in the system. This folder is here to show an example of the images under the label "bottle", with the different folders representing the good and anomalous images. 
2. [Data Loading](#data_loading.py) : This python file represents the code allowing the loading of the MvTec Dataset
3. [PatchCore Architecture](#patchcore_model.py) : This python file contains the code defining the PatchCore class and model architecture
4. [Utils](#utils.py) : This python file contains some methods used in the PatchCore core algorithm
5. [Run method](#run.py) : This python file is the main one to execute the whole program
6. [Run notebook](#patchcore_run_notebook.ipynb) : This notebook allows to run the whole program too and show an exemple of an execution
7. [Neural Networks](#neural_networks.py) : This python file contains code about the architecture of the WideResNet50 and the ResNet neural network. We experiment to implement them to better understand the architecture.

## Instructions of use 

### Requirements 
First, some python requirements are needed, see the following : 
```
wget==3.2
matplotlib==3.3.4
timm==0.4.12
click==7.1.2
torch==1.13.1
numpy==1.19.5
torchvision==0.14.1
Pillow==9.3.0
scikit_learn==0.24.2
```

### Run the program

To run the whole program there are 2 possibilities : 

1. Download all the needed files (data_loading.py, utils.py, patchcore_model.py and run.py). Then you can run directly the run.py file, it is required to have a system configured with GPU, since the computation is heavy. 

2. Download just the 3 following files : data_loading, utils.py, patchcore_model.py and the notebook patchcore_run_model.py. Then you can use directly the notebook to execute the program. The 3 first downloaded files must be accessible in the notebook. The notebook allow to use some additional code to plot and compare the roc_auc score obtained between the different datasets and the different choices made in the model parameter (variation in the value of f_coreset, the percentage of the tensors in the Memory Bank to keep).

In the 2 options, the execution of the code will display the steps of the computation, the roc_auc score obtained at the end of the performance evaluation and a plot representing the ROC curve for the tested dataset. 

For the data loading part, change the variable "DATASETS_PATH" in the data_loading.py file, with the directory of where you locally have the dataset, in case you already have it, or where you want it to be stored, in case you don't have it already.

## Project report  

[See the report pdf](./s306595_s306625_s316359_project5.pdf/)

## Team 
Maélis YONES - [@MaelisYones](https://github.com/MaelisYONES)  
Chloé TEMPO - [@chlotmpo](https://github.com/chlotmpo)  
Niccolo GIOVENALI - [@NiccoloGiovenali](https://github.com/NiccoloGiovenali)
