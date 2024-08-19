import argparse
import os
import json
import torch
import math
import copy
import itertools
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from data import *
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from torch_geometric import data as DATA
from tools.model import *
from sklearn.metrics import accuracy_score, f1_score, auc, precision_recall_curve, average_precision_score, roc_auc_score
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)

def append_file(file_name, data):
    with open(file_name, 'a') as f:
        f.write(str(data)+'\n')
    f.close()

def safemakedirs(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)

def fine_tune(Data, encoder, classifymodel, optimizer, scheduler, drug_folder, kfold_num, param1, param2):
    classification_loss = nn.BCEWithLogitsLoss()
    num_epoch = param2['train_num_epochs']
    best_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0
    }
    best_test_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0
    }
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    
    encoder.eval()
    classifymodel.train()

    # log
    loss_log_name = os.path.join(drug_folder, str(kfold_num) + 'train_loss_log.txt')
    eval_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldeval.txt')
    test_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldtest.txt')
    best_eval_log_name = os.path.join(drug_folder, str(kfold_num) + "_fold_best_auc.txt")

    train_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_trainfeature')
    eval_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_evalfeature')
    test_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_testfeature')
    # # stop
    tolerance = 0
    max_tolerance = 20
    for epoch in range(num_epoch):
        finetune_lossdict = defaultdict(float)
        optimizer.zero_grad()
        # print("train epoch:", epoch)
        z = encoder.encode(source_train_data[0])
        predict = classifymodel(z)
        c_loss = classification_loss(predict, source_train_data[1])
        loss = c_loss
        loss.backward()
        optimizer.step()
        if param1['scheduler_flag'] == True:
            scheduler.step()
        if torch.is_tensor(c_loss):
            c_loss = c_loss.item()
        finetune_lossdict.update({'class_loss': c_loss})
        append_file(loss_log_name, finetune_lossdict)
        # eval and test
        with torch.no_grad():
            evalemb = encoder.encode(source_test_data[0])
            testemb = encoder.encode(target_data[0])
            trainemb = encoder.encode(source_train_data[0])
            
            eval_y_pred = classifymodel(evalemb).cpu().detach().numpy()
            test_y_pred = classifymodel(testemb).cpu().detach().numpy()
            eval_y_true = source_test_data[1].cpu().detach().numpy()
            test_y_true = target_data[1].cpu().detach().numpy()
            # metrics
            eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            eval_auprc = average_precision_score(eval_y_true,eval_y_pred)
            # eval_auprc = auprc(eval_y_true, eval_y_pred)
            eval_f1 = f1_score(eval_y_true, (eval_y_pred > 0.5).astype('int'))
            eval_acc = accuracy_score(eval_y_true, (eval_y_pred > 0.5).astype('int'))
            eval_metrics = {
                'EPOCH:': epoch,
                'AUC': eval_auc,
                'AUPRC': eval_auprc,
                'F1': eval_f1,
                'Accuracy': eval_acc
            }
            append_file(eval_log_name, eval_metrics)
            test_auc = roc_auc_score(test_y_true, test_y_pred)
            test_auprc = average_precision_score(test_y_true,test_y_pred)
            # test_auprc = auprc(test_y_true, test_y_pred)
            test_f1 = f1_score(test_y_true, (test_y_pred > 0.5).astype('int'))
            test_acc = accuracy_score(test_y_true, (test_y_pred > 0.5).astype('int'))
            test_metrics = {
                'EPOCH:': epoch,
                'AUC': test_auc,
                'AUPRC': test_auprc,
                'F1': test_f1,
                'Accuracy': test_acc
            }
            append_file(test_log_name, test_metrics)
            # early stop
            if eval_metrics['AUC'] >= best_metrics['AUC']:
                best_metrics.update(eval_metrics)
                best_metrics['EPOCH'] = epoch
                best_test_metrics.update(test_metrics)
                best_test_metrics['EPOCH'] = epoch
                temp_log = {'epoch': epoch, "eval auc=": eval_metrics['AUC'], "test auc=": test_metrics['AUC']}
                append_file(best_eval_log_name, temp_log)
                tolerance = 0
                best_train_feature = trainemb
                best_eval_feature = evalemb
                best_test_feature = testemb
                # pretrainc1
                best_classifier = copy.deepcopy(classifymodel)
            else:
                tolerance += 1
                if tolerance in (10, 20, 50):
                    append_file(best_eval_log_name, {'early stop': tolerance})
            if tolerance >= max_tolerance:
                # print("Early stopping triggered. Training stopped.")
                # best_eval_log_name = os.path.join(drug_folder, "best_auc.txt")
                # append_file(best_eval_log_name, best_metrics)
                break
    # train_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_trainfeature')
    # eval_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_evalfeature')
    # test_feature_name = os.path.join(drug_folder, str(kfold_num)+'_fold_testfeature')
    torch.save(best_train_feature, train_feature_name)
    torch.save(best_eval_feature, eval_feature_name)
    torch.save(best_test_feature, test_feature_name)
    torch.save(best_classifier.state_dict(), os.path.join(drug_folder, str(kfold_num) + 'fold_classifier.pth'))
    print('{}_fold feature and classifier saved , best_test_auc:{}'.format(kfold_num, best_test_metrics['AUC']))
    return best_test_metrics

def classifier_finetune(parent_folder, drug_list, datatype, outfolder, resultname, otherfolder=None):
    # drug_encoder_dict = drugpth
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    fine_tune_params_grid = {
        'ftlr' : [0.01, 0.001],
        'scheduler_flag' : [True, False]
    }
    ftkeys, ftvalues = zip(*fine_tune_params_grid.items())
    fine_tune_dict_list = [dict(zip(ftkeys, v)) for v in itertools.product(*ftvalues)]
    all_metrics =  {}
    for drug in drug_list:
        all_metrics.update({drug+'auc':0, drug+'AUPRC':0, drug+'folder':None})
    best_df = pd.DataFrame(index=drug_list, columns=['auc', 'aucvar', 'auprc', 'auprcvar'])
    for fine_tune_dict in fine_tune_dict_list:
        for param in update_params_dict_list:
            for drug in drug_list:
                set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                               ',t_epochs_' + str(param['train_num_epochs']) + \
                               ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                               ',tlr' + str(param['gan_learning_rate'])
                model_folder = os.path.join(parent_folder, set_dir_name)
                print(set_dir_name)
                encoder_state_dict = torch.load(os.path.join(model_folder, 'after_traingan_shared_vae.pth'))
                print('train drug:', drug)
                if datatype == 'PDTC':
                    data_generator = PDTC_data_generator(drug)
                elif datatype == 'TCGA':
                    data_generator = TCGA_data_generator(drug)
                else:
                    data_generator = other_data_generator(os.path.join(otherfolder, drug))
                auc_folder = os.path.join(model_folder, 'feature_save')
                drug_auc_folder = os.path.join(auc_folder, drug)
                safemakedirs(drug_auc_folder)
                test_auc_list = []
                i = 0  # fold num
                addauc = []
                addauprc = []
                for data in data_generator:
                    temp_folder = os.path.join(drug_auc_folder,"ftepoch"+str(param['train_num_epochs'])+"_lr_"+str(fine_tune_dict['ftlr'])+"_CosAL_"+str(fine_tune_dict['scheduler_flag']))
                    log_folder = os.path.join(temp_folder, 'log')
                    safemakedirs(log_folder)
                    test_auc_log_name = os.path.join(temp_folder, 'classifier_test_auc.txt')
                    temp_encoder = VAE_mask(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
                    temp_encoder.load_state_dict(encoder_state_dict)
                    classifymodel = Classify(input_dim=32).to(device)
                    fine_tune_optimizer = torch.optim.AdamW(classifymodel.parameters(), lr=fine_tune_dict['ftlr'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fine_tune_optimizer, param['train_num_epochs'])
                    test_history = fine_tune(
                        Data = data,
                        encoder=temp_encoder,
                        classifymodel=classifymodel,
                        optimizer = fine_tune_optimizer,
                        scheduler = scheduler,
                        drug_folder=log_folder,
                        kfold_num = i,
                        param1 = fine_tune_dict,
                        param2 = param
                    )
                    test_auc_list.append(test_history)
                    addauc.append(test_history['AUC'])
                    addauprc.append(test_history['AUPRC'])
                    i=i+1
                    if i==5:
                        meanauc = sum(addauc)/len(addauc)
                        meanauprc = sum(addauprc)/len(addauprc)
                        if meanauc > all_metrics[drug+'auc']:
                            all_metrics[drug+'auc'] = meanauc
                            all_metrics[drug+'AUPRC'] = meanauprc
                            all_metrics[drug+'folder'] = temp_folder
                            best_df.at[drug, 'auc'] = meanauc
                            best_df.at[drug, 'auprc'] = meanauprc
                            best_df.at[drug, 'aucvar'] = np.var(addauc)
                            best_df.at[drug, 'auprcvar'] = np.var(addauprc)
                        print('pretrain mean auc:', sum(addauc)/len(addauc))
                        with open(test_auc_log_name,'w') as f:
                            for item in test_auc_list:
                                f.write(str(item)+'\n')
                        f.close()
    result_path = os.path.join(outfolder, resultname)
    best_df.to_csv(result_path, index=True)                   
    return all_metrics

def main_train_classifier(i):
    pretrain_model = './vae_mask_result/vae_mask_pretrain/pretrain_mask'+str(i)
    outfolder = './vae_mask_result/vae_mask_final_result'
    outname = 'mask_result'+str(i)+'.csv'
    safemakedirs(outfolder)
    # for dataset in ['PDTC', 'TCGA']:
    for dataset in ['TCGA']:
    # for dataset in ['PDTC']:
        if dataset == 'PDTC':
            pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
            drug_list = pdtc_drug_file.index.tolist()
            classifier_metrics = classifier_finetune(pretrain_model, drug_list, dataset, outfolder, 'PDTC_'+outname)
            classifier_file_path = 'PDTC_mask_time'+str(i)+'.txt'
            with open(classifier_file_path, "w") as file:
                file.write(json.dumps(classifier_metrics))
        elif  dataset == 'TCGA':
            drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
            classifier_metrics = classifier_finetune(pretrain_model, drug_list, dataset, outfolder, 'TCGA_'+outname)
            classifier_file_path = 'tcga_mask_time'+str(i)+'.txt'
            with open(classifier_file_path, "w") as file:
                file.write(json.dumps(classifier_metrics))

if __name__ == '__main__':
    for i in range(0, 10):
        parser = argparse.ArgumentParser('train_classifier')
        parser.add_argument('--dataset', dest='dataset', default='TCGA', choices=['TCGA', 'PDTC', 'other'])
        parser.add_argument('--data', dest='data', type=str, default=None, help='data folder, if you use your own dataset(dataset == other) , this folder contains folders,'
                                                                                'that the folders contain sourcedata.csv sourcelabel.csv targetdata.csv targetlabel.csv')
        parser.add_argument('--drug', dest='drug', type=str, default=None, help='contains drugnames')
        parser.add_argument('--pretrain_model', dest='pretrain_model', type=str, default='./vae_mask_result/vae_mask_pretrain/pretrain_mask'+str(i), help='pretrain model folder')
        parser.add_argument('--outfolder', dest='outfolder', type=str, default='./vae_mask_result/vae_mask_final_result', help='folder to save result')
        parser.add_argument('--outname', dest='outname', type=str, default='mask_result'+str(i)+'.csv', help='result .csv file')
        args = parser.parse_args()
        safemakedirs(args.outfolder)
        if args.dataset in ['TCGA', 'PDTC']:
            if args.dataset == 'PDTC':
                pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
                drug_list = pdtc_drug_file.index.tolist()
                classifier_metrics = classifier_finetune(args.pretrain_model, drug_list, args.dataset, args.outfolder, 'PDTC_'+args.outname)
                classifier_file_path = 'pdtc_time'+str(i)+'.txt'
                with open(classifier_file_path, "w") as file:
                    file.write(json.dumps(classifier_metrics))
            elif args.dataset == 'TCGA':
                drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
                classifier_metrics = classifier_finetune(args.pretrain_model, drug_list, args.dataset, args.outfolder, 'TCGA_'+args.outname)
                classifier_file_path = 'tcga_time'+str(i)+'.txt'
                with open(classifier_file_path, "w") as file:
                    file.write(json.dumps(classifier_metrics))
        elif args.dataset == 'other':
            drug_file = pd.read_csv(args.drug, index_col=0)
            drug_list = drug_file.index.tolist()
            classifier_metrics = classifier_finetune(args.pretrain_model, drug_list, args.dataset, args.outfolder, 'other_'+args.outname, args.data)
            classifier_file_path = 'other_time'+str(i)+'.txt'
            with open(classifier_file_path, "w") as file:
                file.write(json.dumps(classifier_metrics))
