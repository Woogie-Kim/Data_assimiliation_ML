import algorithms
from copy import deepcopy
from sklearn.metrics import mean_absolute_percentage_error as mape
from utils import set_newperm
import numpy as np
import torch

class Ensemble_method:
    def __init__(self, args, alg_name='EnKF', num_of_data=None, use_latent=False):
        self.args = args
        self.num_of_APtime = args.num_of_APtime
        if num_of_data:
            self.num_of_points = num_of_data
        else:
            self.num_of_points = args.num_of_Atime
        self.algorithm = getattr(algorithms, alg_name)(args, num_of_data)
        self.perm_list = []
        self.error = []
        self.use_latent = use_latent
        self.device = args.device


    def iterate(self, ens, ref, alpha=1, num_of_points=None, isnew=True, model=None):
        if not num_of_points: num_of_points = self.num_of_points
        ref = deepcopy(ref)

        for run in range(num_of_points):
            # prediction step
            ens = self.algorithm.execute(ens, run)
            ens_m = self.algorithm.set_mean_vector(ens, isnew)
            self.HM_error(ens, ref, run)
            # assimilation step
            updated_perms = self.algorithm.update(ens, ref, ens_m, run, alpha=alpha, use_latent=self.use_latent, model=model)
            updated_perms = self._cut_boundary(updated_perms)
            if self.use_latent:
                for idx, e in enumerate(ens):
                    e.latent = torch.FloatTensor(updated_perms[idx]).view(1,-1).to(self.device)
                ens = set_newperm(ens, model)
            else:
                for idx, e in enumerate(ens):
                    e.perm = updated_perms[idx]
            self.perm_list.append(updated_perms)
        return ens

    def evaluate(self, ens, ref, directory=None):
        ens_eval = deepcopy(ens)
        for e in ens_eval: e.characterization_algorithm='evaluate'
        self.algorithm.execute(ens_eval, 'evaluate', directory=directory)
        ap_error = self.HM_error(ens_eval, ref, iseval=True, aptime=True)
        error = self.HM_error(ens_eval, ref, iseval=True)
        print(f'HM error (%) : {error*1e2:.2f} \nHM error after given data (%) : {ap_error*1e2:.2f}')
        return ens_eval

    def HM_error(self, ens, ref, run=0, iseval=False, aptime=False):
        n_obs = len(ens[0].observation['total'])
        if ens[0].characterization_algorithm != 'EnKF':
            run=0
        true = ref.observation['total'][n_obs*run:n_obs*(run+1)]
        if iseval:
            true = ref.observation['total']
        e_l = []
        for e in ens:
            if aptime: error = mape(true[-self.num_of_APtime:], e.observation['total'][-self.num_of_APtime:])
            else: error = mape(true, e.observation['total'])
            e.error = error
            e_l.append(error)
        self.error.append(np.mean(e_l))
        return self.error[-1]

    def _cut_boundary(self, perms):
        perms[perms >=20000] = 20000
        perms[perms <= 0.01] = 0.01
        return perms

