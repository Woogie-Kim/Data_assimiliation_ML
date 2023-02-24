import pandas as pd
import numpy as np
import os, shutil, subprocess

class Simulate:
    def __init__(self, args):
        self.args = args
        self.ecl_filename = args.ecl_filename
        self.filepath = args.filepath
        self.solution_filename = args.solution_filename
        self.simulation_directory = args.simulation_directory
        self.reference_directory = args.reference_directory
        self.initial_directory = args.initial_directory
        self.assimilation_directory = args.assimilation_directory
        self.mean_directory = args.mean_directory
        self.tstep_filename = args.tstep_filename
        self.stepsize = args.stepsize
        self.num_of_Atime = args.num_of_Atime
        self.num_of_Ptime = args.num_of_Ptime
        self.observation_label = args.observation_label
        self.perm_filename = args.perm_filename
        self.PATH = args.PATH
        self.characterization_algorithm = args.characterization_algorithm
        self.initialize()

    def initialize(self):
        if not os.path.exists(self.simulation_directory):
            os.mkdir(self.simulation_directory)
        if not os.path.exists(f'{self.simulation_directory}/{self.reference_directory}'):
            os.mkdir(f'{self.simulation_directory}/{self.reference_directory}')
        if not os.path.exists(f'{self.simulation_directory}/{self.initial_directory}'):
            os.mkdir(f'{self.simulation_directory}/{self.initial_directory}')
        if not os.path.exists(f'{self.simulation_directory}/{self.assimilation_directory}'):
            os.mkdir(f'{self.simulation_directory}/{self.assimilation_directory}')
        if self.args.isnew:
            if not os.path.exists(f'{self.simulation_directory}/{self.mean_directory}'):
                os.mkdir(f'{self.simulation_directory}/{self.mean_directory}')
    
    def _run_program(self, program, filename, directory):
        command = fr"C:\\ecl\\2009.1\\bin\\pc\\{program}.exe {filename} > NUL"
        os.chdir(os.path.join(self.PATH, self.simulation_directory, directory))
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        os.chdir(self.PATH)
    
    def _get_datafile(self, filename):
        with open(os.path.join(self.PATH,self.filepath, filename + '.DATA'), 'r') as f:
            lines = f.readlines()
        return lines
    
    def _make_permfield(self, directory):
        with open(os.path.join(self.PATH, self.simulation_directory, directory, self.perm_filename + '.DATA'), 'w') as f:
            f.write('PERMX\n')
            for p in self.perm:
                try:
                    f.write(f'{p[0]}\n')
                except:
                    f.write(f'{p}\n')
            f.write('/')

    def _set_datafile(self, filename, rawdata, directory):
        with open(os.path.join(self.PATH, self.simulation_directory, directory, filename + '.DATA'), 'w') as f:
            for line in rawdata:
                if self.perm_filename in line:
                    f.write(f"'{self.perm_filename}.DATA' /\n")
                elif self.tstep_filename in line:
                    f.write(f"'{self.tstep_filename}.DAT' /\n")
                elif self.solution_filename in line:
                    f.write(f"'{self.solution_filename}.DAT' /\n")
                else:
                    f.write(line)
            f.write('\n')
            
    def make_solution(self, directory, run):
        with open(os.path.join(self.PATH,self.simulation_directory,directory, self.solution_filename + '.DAT'), 'w') as f:
            f.write('RESTART\n')
            f.write(f'{self.ecl_filename}_{run} {run+1} /')

    def _set_tstepfile(self, directory, idx):
        with open(os.path.join(self.PATH,self.simulation_directory,directory, self.tstep_filename + '.DAT'), 'w') as f:
            f.write('TSTEP\n')
            f.write(f'{int(idx)}*{int(self.stepsize)}/\n')
            f.close()
    
    def _get_proddata(self, directory, time, run):
        if self.characterization_algorithm == 'EnKF' or self.characterization_algorithm == 'ES_MDA':
            fname = f'{self.ecl_filename}_{run}.RSM'
        else:
            fname = self.ecl_filename + '.RSM'
        with open(os.path.join(self.PATH,self.simulation_directory,directory, fname), 'r') as f:
            lines = f.readlines()

        heads = []
        data = []
        well = []
        for line in lines:
            if 'TIME' in line:
                heads.append(line.split())
            if any(word.isdigit() for word in line) and not 'P' in line:
                line_ = line.split()
                if len(line_) != 1: data.append(line_)
            if 'P' in line and not 'SUMMARY' in line and not 'TIME' in line:
                well.append(line.split())

        length = time + 1
        if self.characterization_algorithm == 'EnKF' and run >= 1:
            length = time
        for idx, head in enumerate(heads):
                if idx==0: df = pd.DataFrame(data=data[idx*length:(idx+1)*length],columns = head).astype('float')
                else: df = pd.concat([df, pd.DataFrame(data=data[idx*length:(idx+1)*length],columns = head).astype('float')], axis=1)



        WOPR = df['WOPR']
        WOPR.columns = well[0]
        WWCT = df['WWCT']
        WWCT.columns = well[1]
        FOPT = df['FOPT']
        FWPT = df['FWPT']
        return WOPR, WWCT, FOPT, FWPT
    
    def eclipse_parallel(self, directory, run):
        data_path = os.path.join(self.PATH, self.filepath)
        path = os.path.join(self.PATH, self.simulation_directory, directory)
        if not os.path.exists(path): os.mkdir(path)
        
        if self.characterization_algorithm == 'EnKF':
            num_of_datapoint = 1
        elif self.characterization_algorithm == 'ES' or self.characterization_algorithm == 'ES_MDA':
            num_of_datapoint = self.num_of_Atime
        else:
            num_of_datapoint = self.num_of_Ptime

        if run == 0 or self.characterization_algorithm != 'EnKF':
            data_path = os.path.join(self.PATH, self.filepath)
            path = os.path.join(self.PATH, self.simulation_directory, directory)
            shutil.copy(os.path.join(data_path, self.solution_filename + '.DAT'), path)    
        else:
            self.make_solution(directory, run-1)

        self._set_tstepfile(directory, num_of_datapoint)
        self._make_permfield(directory)

        if self.characterization_algorithm == 'EnKF' or self.characterization_algorithm == 'ES_MDA':
            ecl_filename = f'{self.ecl_filename}_{run}'
            shutil.copy(os.path.join(data_path, self.ecl_filename + '.DATA'), f'{path}/{ecl_filename}.DATA')
        else:
            ecl_filename = self.ecl_filename
            shutil.copy(os.path.join(data_path, self.ecl_filename + '.DATA'), path)

        datafile_raw = self._get_datafile(self.ecl_filename)
        self._set_datafile(ecl_filename, datafile_raw, directory)
        self._run_program('eclipse', ecl_filename, directory)
    

    def ecl_results(self, directory, run=0):
        if self.characterization_algorithm == 'EnKF':
            num_of_datapoint = 1
        elif self.characterization_algorithm == 'ES' or self.characterization_algorithm =='ES_MDA':
            num_of_datapoint = self.num_of_Atime
        else:
            num_of_datapoint = self.num_of_Ptime

        WOPR, WWCT, FOPT, FWPT = self._get_proddata(directory, num_of_datapoint, run)
        self.results['WOPR'].append(WOPR)
        self.results['WWCT'].append(WWCT)
        self.results['FOPT'].append(FOPT)
        self.results['FWPT'].append(FWPT)

        self.observation['total'] = []
        for label in self.observation_label:
            self.observation[label] = self.results[label][-1]
            self.observation['total'].append(self.observation[label].iloc[-num_of_datapoint:,::].to_numpy().flatten())
        self.observation['total'] = np.concatenate(self.observation['total'])
        return self.observation

