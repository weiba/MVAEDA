import os
import torch
import math
import copy
import itertools
import argparse
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

def append_file(file_name, data):
    with open(file_name, 'a') as f:
        f.write(str(data)+'\n')
    f.close()

def apply_noise(X, p=None):
    if p is None:
        p = [0.4]*1426
    p = torch.tensor(p)
    should_swap = torch.bernoulli(p.to(
        X.device) * torch.ones((X.shape)).to(X.device))
    corrupted_X = torch.where(
        should_swap == 1, X[torch.randperm(X.shape[0])], X)
    masked = (corrupted_X != X).float()
    return corrupted_X, masked

def safemakedirs(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)

def ortho_loss(shared_z, private_z):
    s_l2_norm = torch.norm(shared_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = shared_z.div(s_l2_norm.expand_as(shared_z) + 1e-6)
    p_l2_norm = torch.norm(private_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = private_z.div(p_l2_norm.expand_as(private_z) + 1e-6)
    ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
    return ortho_loss

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.repeat(real_samples.shape[0] // fake_samples.shape[0], 1))).requires_grad_(True)
    # print("inter shape", interpolates.size())
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_discrim(s_batch, t_batch, shared_encoder, sencoder,tencoder, discrim, optimizer,scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.eval()
    tencoder.eval()
    shared_encoder.eval()
    discrim.train()
    optimizer.zero_grad()

    s_batch_mask,smask = apply_noise(s_batch)
    t_batch_mask,tmask = apply_noise(t_batch)
    # pvae
    pzs,_ = sencoder.loss_mask(s_batch_mask, smask)
    pzt,_ = tencoder.loss_mask(t_batch_mask, tmask)
    # vae
    zs,_ = shared_encoder.loss_mask(s_batch, smask)
    zt,_ = shared_encoder.loss_mask(t_batch, tmask)
    # F.normalize(latent_code, p=2, dim=1)
    s = torch.cat((zs, pzs), dim=1)
    t = torch.cat((zt, pzt), dim=1)
    d_loss = torch.mean(t)-torch.mean(s)
    g_p = compute_gradient_penalty(
        critic=discrim,
        real_samples=s,
        fake_samples=t,
        device=device
    )
    loss_log.update({"discrim_loss":d_loss, "g_p":g_p})
    d_loss = d_loss + 10*g_p
    d_loss.backward()
    optimizer.step()
    scheduler.step()
    discrim.eval()

    return loss_log

def train_d_ae(s_batch, t_batch, shared_encoder, sencoder, tencoder, discrim,  optimizer,scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.train()
    tencoder.train()
    shared_encoder.train()
    discrim.eval()
    optimizer.zero_grad()

    s_batch_mask, smask = apply_noise(s_batch)
    t_batch_mask, tmask = apply_noise(t_batch)
    # private
    pccle_z,pccle_vae_loss  = sencoder.loss_mask(s_batch_mask, smask)
    ptcga_z,ptcga_vae_loss  = tencoder.loss_mask(t_batch_mask, tmask)

    # shared
    ccle_z, ccle_vae_loss = shared_encoder.loss_mask(s_batch_mask, smask)
    tcga_z, tcga_vae_loss = shared_encoder.loss_mask(t_batch_mask, tmask)
    # vaeloss
    pvae_loss = pccle_vae_loss+ptcga_vae_loss
    vae_loss = ccle_vae_loss + tcga_vae_loss
    # oloss
    o_loss = ortho_loss(ccle_z, pccle_z)+ortho_loss(tcga_z,ptcga_z)
    # gloss
    g_loss = -torch.mean(discrim(torch.cat((tcga_z, ptcga_z), dim=1)))
    loss = o_loss + + g_loss + vae_loss + pvae_loss
    loss_log.update({"ortho_loss":o_loss, "pvae_loss":pvae_loss, "gen_loss":g_loss, "vae_loss":vae_loss})
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss_log

def pretrain_mask_vae(sourcedata, targetdata, param, parent_folder, times):
    print("start pretrain")
    set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                   ',t_epochs_' + str(param['train_num_epochs']) + \
                   ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                   ',tlr' + str(param['gan_learning_rate'])
    pretrain_dir = os.path.join(parent_folder, set_dir_name)
    safemakedirs(pretrain_dir)
    print(set_dir_name)

    trainloss_logfile = os.path.join(pretrain_dir, "pretrain_losslog.txt")
    evalloss_logfile = os.path.join(pretrain_dir, "pretrain_eval_losslog.txt")

    trainloss_logdict = defaultdict(float)
    evalloss_logdict = defaultdict(float)

    # data
    sourcetrainloader = sourcedata[0]
    sourcetest_nomask = sourcedata[1]
    targettrainloader = targetdata[0]
    targettest_nomask = targetdata[1]
    # shared
    shared_vae = VAE_mask(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    # private
    source_private_vae = VAE_mask(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    target_private_vae = VAE_mask(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    if os.path.exists(os.path.join(pretrain_dir, "shared_vae.pth")) == False:
        # model dict
        source_dict = source_private_vae.state_dict()
        shared_dict = shared_vae.state_dict()
        target_dict = target_private_vae.state_dict()
        # pretrain parameter
        pretrain_epochs = param['pretrain_num_epochs']
        learning_rate = param['pretrain_learning_rate']
        tolerance = 0
        max_tolerance = 50
        min_loss = float('inf')
        models = [shared_vae, source_private_vae, target_private_vae]
        models_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        optimizer = torch.optim.Adam(chain(*models_parameters), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
        for epoch in range(pretrain_epochs):
            if epoch % 20 == 0:
                print("pretrain epoch:", epoch)
            train_epoch_oloss = 0
            train_epoch_vaeloss = 0
            train_epoch_pvaeloss = 0
            # start train
            for model in models:
                model.train()
            # print("train")
            for i, ccledata_nomask in enumerate(sourcetrainloader):
                tcgadata_nomask = next(iter(targettrainloader))
                optimizer.zero_grad()
                ccledata_nomask = ccledata_nomask[0]
                tcgadata_nomask = tcgadata_nomask[0]

                # mask
                ccledata,ccleaddmask = apply_noise(ccledata_nomask)
                tcgadata,tcgaaddmask = apply_noise(tcgadata_nomask)

                pccle_z, pccle_vae_loss = source_private_vae.loss_mask(ccledata, ccleaddmask)
                ptcga_z, ptcga_vae_loss = target_private_vae.loss_mask(tcgadata, tcgaaddmask)

                ccle_z, ccle_vae_loss = shared_vae.loss_mask(ccledata, ccleaddmask)
                tcga_z, tcga_vae_loss = shared_vae.loss_mask(tcgadata, tcgaaddmask)

                p_vae_loss = pccle_vae_loss + ptcga_vae_loss
                vae_loss = ccle_vae_loss + tcga_vae_loss
                o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)

                loss = o_loss + vae_loss + p_vae_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_epoch_oloss += o_loss
                train_epoch_vaeloss += vae_loss
                train_epoch_pvaeloss += p_vae_loss

            for model in models:
                model.eval()
            train_epoch_oloss, train_epoch_pvaeloss, train_epoch_vaeloss = train_epoch_oloss / i, train_epoch_pvaeloss / i, train_epoch_vaeloss / i
            trainloss_logdict.update({
                "epoch": epoch,
                "ortholoss": train_epoch_oloss,
                "pVAE_loss": train_epoch_pvaeloss,
                "VAE_loss": train_epoch_vaeloss
            })
            append_file(trainloss_logfile, trainloss_logdict)
            # print("eval")
            with torch.no_grad():
                sourcetest, sourcetestmask = apply_noise(sourcetest_nomask)
                targettest, targettestmask = apply_noise(targettest_nomask)

                pccle_z, pccle_vae_loss = source_private_vae.loss_mask(sourcetest, sourcetestmask)
                ptcga_z, ptcga_vae_loss = target_private_vae.loss_mask(targettest, targettestmask)

                ccle_z, ccle_vae_loss = shared_vae.loss_mask(sourcetest, sourcetestmask)
                tcga_z, tcga_vae_loss = shared_vae.loss_mask(targettest, targettestmask)

                # eval vae loss
                eval_vae_loss = ccle_vae_loss + tcga_vae_loss
                eval_pvae_loss = pccle_vae_loss + ptcga_vae_loss
                # eval ortho loss
                eval_oloss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)
                evalloss_logdict.update({
                    "epoch": epoch,
                    "ortholoss": eval_oloss,
                    "pVAE_loss": eval_pvae_loss,
                    "VAE_loss": eval_vae_loss
                })
                append_file(evalloss_logfile, evalloss_logdict)
                
                evalloss = eval_oloss + eval_pvae_loss + eval_vae_loss
                if evalloss < min_loss:
                    min_loss = evalloss
                    tolerance = 0
                    source_dict = source_private_vae.state_dict()
                    target_dict = target_private_vae.state_dict()
                    shared_dict = shared_vae.state_dict()
                else:
                    tolerance += 1
                if tolerance >= max_tolerance:
                    print("pretrain early stop")
                    break
        torch.save(shared_dict, os.path.join(pretrain_dir, "shared_vae.pth"))
        torch.save(source_dict, os.path.join(pretrain_dir, "source_vae.pth"))
        torch.save(target_dict, os.path.join(pretrain_dir, "target_vae.pth"))
    else:
        shared_dict = torch.load(os.path.join(pretrain_dir, "shared_vae.pth"))
        source_dict = torch.load(os.path.join(pretrain_dir, "source_vae.pth"))
        target_dict = torch.load(os.path.join(pretrain_dir, "target_vae.pth"))
    # train gan
    if os.path.exists(os.path.join(pretrain_dir, 'after_traingan_shared_vae.pth')) == True:
        print("after train gan model exists")
    elif os.path.exists(os.path.join(pretrain_dir, 'after_traingan_shared_vae.pth')) == False:
        print("start gan train")
        gan_epoch = param['train_num_epochs']
        gan_lr = param['gan_learning_rate']
        shared_vae.load_state_dict(shared_dict)
        source_private_vae.load_state_dict(source_dict)
        target_private_vae.load_state_dict(target_dict)
        d_ae_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        discrim = Discriminator(input_dim=32 + 32).to(device)
        discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
        discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, gan_epoch)
        d_ae_optimizer = torch.optim.RMSprop(chain(*d_ae_parameters), lr=gan_lr)
        d_ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_ae_optimizer, gan_epoch)
        dloss_logfile = os.path.join(pretrain_dir, "d_losslog.txt")
        genloss_logfile = os.path.join(pretrain_dir, "g_losslog.txt")
        max_gan_tolerance = 20
        gan_tolerance = 0
        shared_vae_aftergan_dict = shared_vae.state_dict()
        min_loss = float('inf')
        for epoch in range(gan_epoch):
            temp_loss = 0
            dloss_list = []
            genloss_list = []
            if epoch % 10 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for step, ccledata in enumerate(sourcetrainloader):
                tcgadata = next(iter(targettrainloader))
                dlosslog = train_discrim(s_batch=ccledata[0],
                                         t_batch=tcgadata[0],
                                         shared_encoder=shared_vae,
                                         sencoder=source_private_vae,
                                         tencoder=target_private_vae,
                                         discrim=discrim,
                                         optimizer=discrim_optimizer,
                                         scheduler=discrim_scheduler
                                         )
                dloss_list.append(dlosslog)
                if (step + 1) % 5 == 0:
                    genlosslog = train_d_ae(s_batch=ccledata[0],
                                            t_batch=tcgadata[0],
                                            shared_encoder=shared_vae,
                                            sencoder=source_private_vae,
                                            tencoder=target_private_vae,
                                            discrim=discrim,
                                            optimizer=d_ae_optimizer,
                                            scheduler=d_ae_scheduler
                                            )
                    genloss_list.append(genlosslog)
            # dloss
            dloss_sum = defaultdict(float)
            dloss_mean = defaultdict(float)
            for metric_dict in dloss_list:
                for metric, value in metric_dict.items():
                    dloss_sum[metric] += value
            num_dicts = len(dloss_list)
            for metric, total_value in dloss_sum.items():
                average_value = total_value / num_dicts
                dloss_mean[metric] = average_value
            # genloss
            genloss_sum = defaultdict(float)
            genloss_mean = defaultdict(float)
            for metric_dict in genloss_list:
                for metric, value in metric_dict.items():
                    genloss_sum[metric] += value
            num_dicts = len(genloss_list)
            for metric, total_value in genloss_sum.items():
                average_value = total_value / num_dicts
                genloss_mean[metric] = average_value
            append_file(dloss_logfile, dloss_mean)
            append_file(genloss_logfile, genloss_mean)
            for key in dloss_mean:
                temp_loss = temp_loss + dloss_mean[key].item()
            for key in genloss_mean:
                temp_loss = temp_loss + genloss_mean[key].item()
            if min_loss > temp_loss:
                gan_tolerance = 0
                shared_vae_aftergan_dict = shared_vae.state_dict()
                min_loss = temp_loss
            else:
                gan_tolerance += 1
            if gan_tolerance >= max_gan_tolerance:
                print("train gan early stop in epoch:", epoch)
                break
        torch.save(shared_vae_aftergan_dict, os.path.join(pretrain_dir, "after_traingan_shared_vae.pth"))

def main_pretrain(i):
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sourcepretrain, targetpretrain = pretrain_data()
    pretrain_path = os.path.join('./vae_mask_result/vae_mask_pretrain','pretrain_mask'+str(i))
    safemakedirs(pretrain_path)
    for param_dict in update_params_dict_list:
        pretrain_mask_vae(sourcepretrain, targetpretrain, param=param_dict, parent_folder=pretrain_path, times = i)

if __name__ == '__main__':
    for i in range(0,10):
        parser = argparse.ArgumentParser('pretrain_mask_vae')
        parser.add_argument('--outfolder', dest='outfolder', default='./vae_mask_result/vae_mask_pretrain-1/pretrain_mask'+str(i), type=str, help='choose the output folder')
        parser.add_argument('--source', dest='source', default=None, type=str, help='.csv file address for the source')
        parser.add_argument('--target', dest='target', default=None, type=str, help='.csv file address for the target')
        args = parser.parse_args()
        params_grid = {
            "pretrain_num_epochs": [0, 100, 300],
            'pretrain_learning_rate': [0.001],
            'gan_learning_rate': [0.001],
            "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
        }
        keys, values = zip(*params_grid.items())
        update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        safemakedirs(args.outfolder)
        if args.source and args.target:
            sourcepretrain = pd.read_csv(args.source, index_col=0, header=0)
            targetpretrain = pd.read_csv(args.target, index_col=0, header=0)
            sourcepretrain = pretrain_loader(sourcepretrain)
            targetpretrain = pretrain_loader(targetpretrain)
            for param_dict in update_params_dict_list:
                pretrain_mask_vae(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, times = i)
        elif args.source and args.target==None:
            sourcepretrain = pd.read_csv(args.source, index_col=0, header=0)
            sourcepretrain = pretrain_loader(sourcepretrain)
            _, targetpretrain = pretrain_data()
            for param_dict in update_params_dict_list:
                pretrain_mask_vae(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, times = i)
        elif args.source == None and args.target:
            sourcepretrain, _ = pretrain_data()
            targetpretrain = pd.read_csv(args.target, index_col=0, header=0)
            targetpretrain = pretrain_loader(targetpretrain)
            for param_dict in update_params_dict_list:
                pretrain_mask_vae(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, times = i)
        else:
            sourcepretrain, targetpretrain = pretrain_data()
            for param_dict in update_params_dict_list:
                pretrain_mask_vae(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, times = i)

