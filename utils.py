import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from dlmodel import ConversionDataset
from torch.utils.data import DataLoader
from glob import glob
import os

def get_parameters(file, directory, BoolofChannel, k_sand, k_clay):
        with open(f'{directory}/{file}', 'r') as f:
            txt = f.read()
            lines = txt.split('\n')
            total_text = [line for line in lines]
        En_num = int(total_text[1])
        En_name_ = total_text[2:En_num+2]
        En_name = [re.findall(r'\d+',En_name_[i]) for i in range(En_num)]
        En_name = np.array([ii[0] for ii in En_name], dtype=int)

        facies = []
        for grid in total_text[En_num+2:-1]:
            facies.append(list(map(float, grid.strip().split(' '))))
        
        facies = np.array(facies)
        if BoolofChannel:
            facies[facies >= 1] = k_sand
            facies[facies < 1] = k_clay
        else:
            facies = np.exp(facies)
        parameters = pd.DataFrame(data=facies, columns=En_name)

        for kk in range(int(total_text[1])):
            if kk == 0:
                data = parameters[kk].to_numpy()
            else:
                data = np.column_stack((data, parameters[kk].to_numpy()))
            
        parameters = pd.DataFrame(data = data, columns= np.linspace(0,int(total_text[1])-1,int(total_text[1]), dtype=int))

        return parameters

def draw_perm(e,args):
    nx = args.nx
    ny = args.ny
    if isinstance(e, list):
        for idx, e_ind in enumerate(e):
            if idx == 0:
                perms = e_ind.perm
            else:
                perms = pd.concat((perms, e_ind.perm), axis=1)
        sns.heatmap(np.log(perms.mean(axis=1).to_numpy().reshape(nx, ny)), cmap='jet', vmax=np.log(2000), vmin=np.log(20))
    else:
        sns.heatmap(np.log(e.perm.to_numpy().reshape(nx,ny)), cmap='jet', vmax= np.log(2000), vmin= np.log(20))

def draw_label(ens, ref, label, args):
    rst = {label: []}
    for e in ens:
        rst[label].append(e.results[label][-1])
    rst = deepcopy(rst)
    for idx, l in enumerate(rst[label]):
        if idx == 0:
            total = l
        else:
            total += l
    ens_m = total /len(ens)
    fig = plt.figure(figsize=[18,12])
    ax=fig.subplots(3,3)

    for i in range(args.num_of_observed_well):
        ax[i // 3, i % 3].plot([0] + list(args.Ptime), ens_m[f'P{i + 1}'].to_numpy(), c='blue')
        ax[i // 3, i % 3].plot([0] + list(args.Ptime), ref.results[label][-1][f'P{i + 1}'].to_numpy(), c='red')
        for e in ens:
            ax[i//3,i%3].plot([0] + list(args.Ptime),e.results[label][-1][f'P{i+1}'].to_numpy(), c=[0.75,0.75,0.75])
        ax[i//3,i%3].plot([0] + list(args.Ptime),ref.results[label][-1][f'P{i+1}'].to_numpy(), c='red')
        ax[i//3,i%3].plot([0] + list(args.Ptime),ens_m[f'P{i+1}'].to_numpy(), c='blue')
        ax[i//3,i%3].set_title(f'P_{i+1}')
        ax[i//3,i%3].axvline(args.history_time, linestyle='--')
        ax[i // 3, i % 3].set_xlim(0, args.Ptime[-1])
    ax[2,2].axis('off')
    fig.legend(["Mean of ensemble models", "Reference", "Ensemble models (400)"],
    loc = 'lower center',
    ncol = 3,
    fontsize = 15,
    fancybox = True,
    framealpha = 1
    )

    plt.show()

def draw_field_label(ens, ref, label, args):
    rst = {label: []}
    for e in ens:
        rst[label].append(e.results[label][-1])
    rst = deepcopy(rst)
    for idx, l in enumerate(rst[label]):
        if idx == 0:
            total = l
        else:
            total += l
    ens_m = total / len(ens)

    fig = plt.figure(figsize=[12,8])
    plt.plot([0] + list(args.Ptime), ens_m.to_numpy(), color='b')
    plt.plot([0] + list(args.Ptime), ref.results[label][-1].to_numpy(), color='r')
    for e in ens:
        plt.plot([0] + list(args.Ptime), e.results[label][-1].to_numpy(), color=[0.75,0.75,0.75])
    plt.plot([0] + list(args.Ptime), ens_m.to_numpy(), color = 'b')
    plt.plot([0] + list(args.Ptime), ref.results[label][-1].to_numpy(), color = 'r')
    plt.axvline(args.history_time, linestyle='--')
    plt.xlim(0, args.Ptime[-1])
    fig.legend(("Mean of ensemble models", "Reference", "Ensemble models (400)"),
    loc = 'lower center',
    ncol = 3,
    fontsize = 15
    )
    plt.show()


def set_latentvec(ens, mdl, args):
    nx = args.nx
    ny = args.ny
    if not isinstance(ens, list): ens = [ens]
    data = mdl.preprocess(ens)
    dataset = ConversionDataset(data, None, nx=nx, ny=ny)
    dataloader = DataLoader(dataset,len(ens),shuffle=False)

    z = mdl.verify(mdl.model, dataloader)

    for en, l in zip(ens, z):
        en.latent = l.view(1,-1)
    return ens

def set_newperm(ens, mdl):
    sc = mdl.scaler
    decoder = mdl.model.Decoder

    if not isinstance(ens, list): ens = [ens]
    for idx, en in enumerate(ens):
        rst = sc.inverse_transform(decoder(en.latent).view(1, -1).detach().cpu().numpy()).flatten()
        rst = mdl._adjust_scaler(rst)
        en.perm = pd.Series(rst, name=idx, dtype=np.float64)
    return ens

def draw_log(mdl):
    typo = 'Times New Roman'
    plt.figure(figsize=(12,5))
    plt.plot(mdl.train_log)
    plt.plot(mdl.valid_log)
    plt.xlabel('Epoch', fontname=typo)
    plt.ylabel('Loss (KL Divergence + Reconstruction Error)', fontname=typo)
    plt.legend({'Train', 'Valid'})
    plt.show()

def remove_rsm(target,args):
    if len(glob(f"./{args.simulation_directory}/{target}/{target}")) == 0:
        [os.remove(f) for f in glob(f"./{args.simulation_directory}/{target}*/*.RSM")]
    else:
        [os.remove(f) for f in glob(f"./{args.simulation_directory}/{target}/{target}*/*.RSM")]

def clear_rsm(args):
    lst = [args.assimilation_directory, args.initial_directory, args.mean_directory, args.reference_directory]
    for rsm in lst:
        remove_rsm(rsm, args)
    print('RSM clear')  