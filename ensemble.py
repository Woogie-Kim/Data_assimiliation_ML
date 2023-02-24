
from simulate import Simulate
class Ensemble(Simulate):
    def __init__(self, args, perm):
        super(Ensemble, self).__init__(args)
        self.args = args
        self.observation_label = args.observation_label
        self.perm = perm
        self.attributes = {'parameter': ['prior', 'assimilated'], 'observation': self.observation_label}
    
        self.parameter = {'prior': [] , 'assimilated': []}
        self.observation = {label: [] for label in self.observation_label}
        self.observation['total'] = []
        self.results = {'WOPR': [], 'WWCT': [], 'FOPT': [], 'FWPT': []}