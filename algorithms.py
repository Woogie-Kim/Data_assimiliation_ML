from tqdm.auto import tqdm
from multiprocessing import Process
from time import sleep

import pandas as pd
import numpy as np
import cupy as cp
from math import *

import os
from copy import deepcopy

from ensemble import Ensemble
from utils import set_latentvec



class EnKF:
    def __init__(self, args, generative=False, num_of_data=None):
        self.args = args
        self.PATH = args.PATH
        self.observation_label = args.observation_label
        self.simulation_directory = args.simulation_directory
        self.assimilation_directory = args.assimilation_directory
        self.initial_directory = args.initial_directory
        self.parallel = args.parallel
        self.generative = generative
        if num_of_data: self.num_of_points = num_of_data
        else: self.num_of_points = args.num_of_Atime
        self.assimilation_path = os.path.join(args.PATH, args.simulation_directory, args.assimilation_directory)


    def update(self, ens, ref, ens_m, run, noise = 0.01, alpha=1, use_latent=False, model=None):
        vec = self._get_vector(ens, use_latent=use_latent,model=model)
        vec_m = self._get_vector(ens_m, use_latent=use_latent,model=model)

        n_obs = len(ens[0].observation['total'])
        if ens[0].characterization_algorithm != 'EnKF':
            run = 0

        obs_r = cp.asarray(ref.observation['total'][n_obs*(run):n_obs *(run+1)])
        cols, n_real = vec.shape

        H = cp.zeros((n_obs, cols))
        for i in range(n_obs): H[-1 -i, -1 -i] = 1

        L = cp.asarray(vec - vec_m)
        cov_e = (1/(n_real-1)) * L@L.T

        Observation =  obs_r.reshape(n_obs,-1) * cp.ones([n_obs, n_real]) + sqrt(alpha) * noise * cp.random.randn(n_obs, n_real)
        C_d = cp.identity(n_obs) * pow(noise, 2)

        K = cov_e @ H.T @ cp.linalg.inv(H @ cov_e @ H.T + alpha * C_d)

        updated_vec = vec + K @ (Observation - H @ vec)
        updated_k = cp.exp(updated_vec[:-n_obs,::])
        return pd.DataFrame(updated_k.get())

    def execute(self, ens, run, directory=None):
        args = self.args
        max_process = args.max_process
        ps = []
        iter_bar = tqdm(ens)
        desc = f'Aissimilation No.{run} '
        if not directory:
            directory = self.assimilation_directory
        if not isinstance(run, int):
            desc = 'Evaluate'
            run=0
        iter_bar.set_description(desc)
        if self.parallel:
            for idx, ensemble in enumerate(iter_bar):
                pth = os.path.join(directory, f'{directory}_{idx}')
                ps.append(Process(target=ensemble.eclipse_parallel, args=(pth, run)))
                if len(ps) == max_process or (idx + 1) == len(ens):
                    for p in ps: p.start()
                    for p in ps: p.join()
                    ps=[]
            sleep(1)
            for idx, ensemble in enumerate(ens): 
                pth = os.path.join(directory, f'{directory}_{idx}')
                ensemble.ecl_results(pth, run)

        else: 
            for idx, ensemble in enumerate(iter_bar):
                pth = os.path.join(directory, f'{directory}_{idx}')
                ensemble.eclipse_parallel(pth, run)
                ensemble.ecl_results(pth, run)
        return ens

    def set_mean_vector(self, ens, isnew=True):
        args = self.args
        args.characterization_algorithm = ens[0].characterization_algorithm
        obs = {label:[] for label in self.observation_label}
        obs['total'] = []
        ens = deepcopy(ens)


        for i, e in enumerate(ens):
            if i==0:
                perm = e.perm
            else:
                perm = pd.concat([perm, e.perm], axis=1)

        perm_m = perm.mean(axis=1)
        ensemble_m = Ensemble(args, perm_m)


        if isnew:
            ensemble_m.eclipse_parallel(args.mean_directory, 0)
            ensemble_m.ecl_results(args.mean_directory)
        else:
            for i, e in enumerate(ens):
                if i == 0:
                    for label in self.observation_label:
                        obs[label] = e.observation[label]
                    obs['total'] = e.observation['total']
                else:
                    for label in self.observation_label:
                        obs[label] += e.observation[label]
                    obs['total'] += e.observation['total']

            for label in self.observation_label:
                ensemble_m.observation[label] = obs[label] / len(ens)
            ensemble_m.observation['total'] = obs['total'] / len(ens)

        # for label in self.observation_label:
        #             obs[label] = obs[label].mean()
        self.mean = ensemble_m
        return ensemble_m


    def _get_vector(self, ens, use_latent=False, model=None):
        if use_latent: set_latentvec(ens, model, self.args)
        obs = []
        if isinstance(ens, list):
            for i, e in enumerate(ens):
                if not use_latent:
                    if i == 0: static = e.perm
                    else: static = pd.concat((static, e.perm),axis=1)
                else:
                    if i == 0: static = pd.Series(e.latent.flatten().detach().cpu())
                    else: static = pd.concat((static, pd.Series(e.latent.flatten().detach().cpu(), name=i)),axis=1)
                obs.append(e.observation['total'])
        else:
            if not use_latent:
                static = ens.perm
            else: static = pd.Series(ens.latent.flatten().detach().cpu())
            obs.append(ens.observation['total'])

        if not use_latent: static = np.log(static)
        return cp.asarray(pd.concat([static, pd.DataFrame(obs).T],axis=0))

        