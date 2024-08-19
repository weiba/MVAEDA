MVAEDA
===============================
Source code and data for "Domain Alignment Method Based on Masked Variational Autoencoder for Predicting Patient Anticancer Drug Response"

# Requirements
All implementations of MVAEDA are based on PyTorch. MVAEDA requires the following dependencies:
- python==3.7.16
- pytorch==1.13.1
- torch_geometric==2.3.1
- numpy==1.21.5+mkl
- scipy==1.7.3
- pandas==1.3.5
- scikit-learn=1.0.2
- hickle==5.0.2
# Data
- Data defines the data used by the model
    - data/TCGA records training data, test data, and labeling related to the five drugs associated with TCGA.
    - data/PDTC records training data, test data, and labeling related to the fifty drugs associated with PDTC.
    - data/ccle_sample_info.csv records biological information related to CCLE samples.
    - data/pretrain_ccle.csv records gene expression data from unlabeled CCLE samples.
    - data/pretrain_tcga.csv records gene expression data from unlabeled TCGA samples.
    - data/pdtc_uq1000_feature.csv records gene expression data from unlabeled PDTC samples.
    - data/GDSC1_fitted_dose_response_25Feb20.csv and data/GDSC2_fitted_dose_response_25Feb20.csv records data on drug use and response in GDSC samples.
    - data/DrugResponsesAUCModels.txt records response data for PDTC sample-drug pairs. 
    - data/pdtc_gdsc_drug_mapping.csv records the 50 drug names associated with pdtc and their smiles.
    - data/uq1000_feature.csv records gene expression data for unlabeled TCGA samples and CCLE samples.
    - data/xena_sample_info_df.csv records biological information related to TCGA samples.
- tools/model.py defines the model used in the training process.
- data.py defines the data loading of the model.
- pretrain_mask_vae.py defines the pre-training of the model.
- classifier.py defines the classifier training of the model.

## Preprocessing your own data
Explanations on how you can process your own data and prepare it for MVAEDA running.
> In our study, the source cell line data and targe partient data we followed are from [codeae](https://codeocean.com/capsule/1993810/tree/v1)[1]. You can run our program with your own data, and process the data you use into source and target domain data of the same dimensions, while just having a one-dimensional labeled data for each sample-drug pair data. You can refer to the style of the data in data/TCGA.
> 
> [1] He, Di, et al. "A context-aware deconfounding autoencoder for robust prediction of personalized clinical drug response from cell-line compound screening." Nature Machine Intelligence 4.10 (2022): 879-892.

# Usage
Once you have configured the environment, you can simply run **MVAEDA** in 2 steps using the data we provided:
```
1. python pretrain_mask_vae.py
2. python classifier.py
```
Or you can just run the following line of code in place of the two above to replicate our experimental results using our data:
```
python train_all.py
```
To bulid and evaluate our model, we uses cell line gene expression data as the source domain and patient gene expression data as the target domain. We divide the source domain data into five folds for cross-validation, with a training set and validation set ratio of 4:1, and the test set uses the target domain data. We use grid search to determine the best parameters and retain the model with the best performance on the validation set in each fold. We also evaluate the classifier performance on each test set. Finally, we use the average AUC and AUPRC of the five five-folds on the test set as our single-run metrics, and then take the average of 10 runs as the final metric.

>In **pretrain_mask_vae.py** We use a cosine annealing strategy with a learning rate set to 0.001, and train epochs by grid searching to store models at different pre-training epochs. We use an early stopping strategy for the model when the validation set loss does not decrease for 50 consecutive times without generative adversarial training and when the validation set loss does not decrease for 20 consecutive times with generative adversarial training.

> In **classifier.py** We train the classifier at this stage by performing a parameter search, testing for the use of a cosine annealing strategy while lr=[0.01, 0.001], Use the early stopping strategy when the AUC value on the validation set does not increase for 20 consecutive times. We predict model performance at this stage.
>

Alternatively, you can run our program with your own data and some other settings as follows:
```
1. python pretrain_mask_vae.py \
--outfolder path/to/folder_to_save_pretrain_models \
--source path/to/your_pretrain_source_data.csv \
--target path/to/your_pretrain_target_data.csv

2. python classifier.py \
--dataset other \
--data path/to/your_data_folder \
--drug path/to/your_drug_name.csv \
--pretrain_model path/to/your_pretrain_models_path \
--outfolder path/to/save_result_and_others \
--outname result_file_name.csv 
```
Note: 
>You need to ensure that the data dimensions of your source and target domains are the same.

> The **your_data_folder** is a folder that contains many medication folders while each medication folder contains sourcedata.csv, targetdata.csv, sourcelabel.csv, targetlabel.csv. The format of each file can be referred to. /data/TCGA.

# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).

