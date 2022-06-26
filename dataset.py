import gzip
from collections import OrderedDict
import os, math
from util import *
import numpy as np

class BaseDataset():
    def getDerandomizationRi(self): 
        sqrt_lambdas = [0] * self.m
        for (id, tot) in self.id_to_tot_articles.items():
            assert(tot > 0) #we don't want empty blogs
            sqrt_lambdas[id] = math.sqrt(tot/self.T)
        
        sm = sum(sqrt_lambdas)
        ris = []
        for id in range(self.m):
            ri = power2_upperbound(sm / sqrt_lambdas[id])
            assert(2**(ri-1) < sm/sqrt_lambdas[id] <= 2**ri)
            ris.append(ri)
        return ris
    
    def getDerandomizationRho(self):   
        return max(self.getDerandomizationRi())

class Dataset(BaseDataset):
    '''
    dataset from Reddit platform
    '''
    def __init__(self, path, discr_step=1, max_num_files = 31, min_number_of_arrivals = 10, max_timestamp = None, min_timestamp = None):
        self.name_to_id = {}
        self.id_to_name = {}
        self.id_to_timestamps = {}
        self.id_to_tot_articles = {}

        self.path = path
        self.discr_step = discr_step
        self.max_num_files = max_num_files
        self.min_number_of_arrivals = min_number_of_arrivals
        self.max_timestamp = max_timestamp
        self.min_timestamp = min_timestamp

        #key: round number, value: dict {key: blog id, value: #articles uploaded}
        self.sparse_matrix = {}

        self.T = 0
        self.m = 0

        id = 0
        self.min_ts = -1
        self.max_ts = -1
        self.min_uploads = -1
        self.max_uploads = -1
        self.avg_uploads = 0
        self.tot_uploaded_articles = 0
        
        assert(os.path.exists(path))
        files = []
        if os.path.isfile(path):
            files = [path]
        else:
            files = sorted(os.listdir(path))
            if max_num_files is not None:
                files = files[:max_num_files]
            files = [path+'/'+f for f in files]

        for path in files:
            with gzip.open(path, 'rt') as f:
                for line in f:
                    #name timestamp_1, ..., timestamp_n
                    sp = line.split()
                    name = sp[0]
                    timestamps = list(map(lambda v: v//discr_step, map(int, sp[1:])))
                    if max_timestamp is not None:
                        timestamps = [t for t in timestamps if t <= max_timestamp]
                    if min_timestamp is not None:
                        timestamps = [t for t in timestamps if t >= min_timestamp]

                    if len(timestamps) < min_number_of_arrivals:
                        continue

                    self.max_ts = max(self.max_ts, max(timestamps))
                    self.min_ts = min(timestamps) if self.min_ts == -1 else min(self.min_ts, min(timestamps))
                    self.max_uploads = max(self.max_uploads, len(timestamps))
                    self.min_uploads = len(timestamps) if self.min_uploads == -1 else min(self.min_uploads, len(timestamps))
                    self.avg_uploads += len(timestamps)
                    self.tot_uploaded_articles += len(timestamps)

                    if name not in self.name_to_id:
                        self.name_to_id[name] = id
                        self.id_to_name[id] = name
                        self.id_to_timestamps[id] = timestamps
                        self.id_to_tot_articles[id] = len(timestamps)
                        id += 1
                    else:
                        #names must be unique
                        assert(False)

        for (id, timestamps) in self.id_to_timestamps.items():
            for ts in timestamps:
                if ts-self.min_ts not in self.sparse_matrix:
                    self.sparse_matrix[ts-self.min_ts] = {}
                if id not in self.sparse_matrix[ts-self.min_ts]:
                    self.sparse_matrix[ts-self.min_ts][id] = 0
                self.sparse_matrix[ts-self.min_ts][id] += 1
        self.T = self.max_ts - self.min_ts
        self.m = len(self.id_to_timestamps)


        self.avg_uploads_per_round = 0
        self.min_uploads_per_round = -1
        self.max_uploads_per_round = -1
        for (_, b) in self.sparse_matrix.items():
            self.avg_uploads_per_round += len(b) #we are actually counting number of blogs non empty per round
            self.max_uploads_per_round = max(self.max_uploads_per_round, len(b))
            self.min_uploads_per_round = len(b) if self.min_uploads_per_round == -1 else min(self.min_uploads_per_round, len(b))           
        
  
        self.avg_uploads /= len(self.id_to_name)
        self.avg_uploads_per_round /= len(self.sparse_matrix)

        print('minimum timestamp:', self.min_ts)
        print('maximum timestamp:', self.max_ts)
        print('duration: ', self.max_ts - self.min_ts + 1)
        print('number of uploaded articles: ', self.tot_uploaded_articles)
        print('min number of uploads per blog: ', self.min_uploads)
        print('max number of uploads per blog: ', self.max_uploads)
        print('avg number of uploads per blog: ', self.avg_uploads)
        print('min uploads per round:', self.min_uploads_per_round)
        print('max uploads per round: ', self.max_uploads_per_round)
        print('avg uploads per round:', self.avg_uploads_per_round)
        print('number of empty rounds: ', self.max_ts - self.min_ts + 1 - len(self.sparse_matrix))
        print('number of non-empty rounds: ', len(self.sparse_matrix))
        print('number of blogs:', len(self.id_to_name))

    def get_dict_state(self):
        return {'path': self.path, 'discr_step':self.discr_step, 'max_num_files': self.max_num_files, 'min_number_of_arrivals':self.min_number_of_arrivals, 
            'max_timestamp': self.max_timestamp, 'min_ts': self.min_ts, 'max_ts': self.max_ts, 'min_uploads_per_blog': self.min_uploads, 
            'max_uploads_per_blog': self.max_uploads,  'avg_uploads_per_blog': self.avg_uploads, 'min_uploads_per_round': self.min_uploads_per_round, 
            'max_uploads_per_round': self.max_uploads_per_round, 'avg_uploads_per_round': self.avg_uploads_per_round, 'non_empty_rounds': len(self.sparse_matrix), 
            'tot uploaded articles': self.tot_uploaded_articles, 'min_timestamp': self.min_timestamp}


class PoissonDataset(BaseDataset):
    def __init__(self, lambdas, T, seed=55, min_l=0.1, max_l=10, lambda_sample='uniform'):
        self.seed = seed
        self.lambda_sample = lambda_sample
        self.min_l = min_l
        self.max_l = max_l
        np.random.seed(seed)
        self.sparse_matrix = {}
        self.id_to_tot_articles = {}
        self.lambdas = lambdas
        if not isinstance(lambdas, list):
            tmp = []
            for i in range(lambdas):
                if lambda_sample == 'uniform':
                    tmp.append(np.random.random()*(max_l - min_l) + min_l)
                elif lambda_sample == 'exponential':
                    tmp.append(min(1, np.random.exponential(scale=1.5)/6)*(max_l - min_l) + min_l)
            self.lambdas = tmp
        self.m = len(self.lambdas)
        self.build_matrix(T)
        print('T:', self.T)
    
    def build_matrix(self, T):
        mxT = -1
        for i in range(self.m):
            self.id_to_tot_articles[i] = 0
            for t in range(T):
                num_articles = np.random.poisson(lam=self.lambdas[i])
                if num_articles == 0 and t == T-1 and self.id_to_tot_articles[i] == 0: #we must make at least an upload (we don't want empty blogs)
                    num_articles = 1
                if num_articles > 0:
                    mxT = max(mxT, t)
                    if t not in self.sparse_matrix:
                        self.sparse_matrix[t] = {}
                    self.sparse_matrix[t][i] = num_articles
                    self.id_to_tot_articles[i] += num_articles
        self.T = mxT
    
    def __str__(self):
        return 'PoissonDataset'

    def get_dict_state(self):
        return {'seed': self.seed, 'T': self.T, 'm': self.m, 'name': self.__str__(), 'min_l': self.min_l, 'max_l': self.max_l, 
                'min_lambda': min(self.lambdas), 'max_lambda': max(self.lambdas), 'lambdas': self.lambdas, 'lambda_sample': self.lambda_sample}

class RenewalProcessDataset(PoissonDataset):
    def __init__(self, lambdas, T, seed=55, min_l=2.001, max_l=4, inter_arrival='zeta', lambda_sample='uniform'):
        self.inter_arrival = inter_arrival
        super().__init__(lambdas, T, seed, min_l, max_l, lambda_sample=lambda_sample)

    def build_matrix(self, T):
        mxT = -1
        actT = [0] * self.m
        ids = OrderedDict([(i, 0) for i in range(self.m)])

        while len(ids) > 0:
            to_rem = []
            for i in ids:
                if i not in self.id_to_tot_articles:
                    self.id_to_tot_articles[i] = 0
                assert(actT[i] < T)

                if self.inter_arrival == 'zeta':
                    next_upload = np.random.zipf(self.lambdas[i])
                else:
                    assert(False)
                actT[i] += next_upload
                if actT[i] > T and self.id_to_tot_articles[i] == 0:
                    actT[i] = T
                if actT[i] <= T:
                    mxT = max(mxT, actT[i])
                    if actT[i] not in self.sparse_matrix:
                        self.sparse_matrix[actT[i]] = {}
                    self.sparse_matrix[actT[i]][i] = 1
                    self.id_to_tot_articles[i] += 1
                if actT[i] >= T:
                    to_rem += [i]
            for i in to_rem:
                del ids[i]

        self.T = mxT

    def __str__(self):
        return 'RenewalProcessDataset'

    def get_dict_state(self):
        d = super().get_dict_state()
        d['inter_arrival'] = self.inter_arrival
        return d

class MarkovChainDataset(BaseDataset):
    def __init__(self, T, m, min_lambdas=[0.8, 0.05], max_lambdas=[1.0, 0.15], change_state_min_prob=[0.05, 0.05], 
                change_state_max_prob=[0.1, 0.1], init_state = 'random', seed=58):
        self.seed = seed
        np.random.seed(seed)
        self.min_lambdas = min_lambdas
        self.max_lambdas = max_lambdas
        self.change_state_min_prob = change_state_min_prob
        self.change_state_max_prob = change_state_max_prob
        self.sparse_matrix = {}
        self.id_to_tot_articles = {}
        self.m = m

        self.lambdas = [[], []]
        self.changeStateP = [[], []]
        for i in range(self.m):
            for j in range(len(min_lambdas)):
                self.lambdas[j].append(np.random.random() * (max_lambdas[j] - min_lambdas[j]) + min_lambdas[j])
                self.changeStateP[j].append(np.random.random() * (change_state_max_prob[j] - change_state_min_prob[j]) + change_state_min_prob[j])

        mxT = 0
        for i in range(m):
            self.id_to_tot_articles[i] = 0
            actual_state = init_state
            if init_state == 'random':
                actual_state = np.random.randint(0, 2)
            for t in range(T):
                coin = np.random.random()
                if coin <= self.changeStateP[actual_state][i]:
                    actual_state = 1 - actual_state
                num_uploads = np.random.poisson(lam=self.lambdas[actual_state][i])
                if num_uploads > 0:
                    if t not in self.sparse_matrix:
                        self.sparse_matrix[t] = {}
                    self.sparse_matrix[t][i] = num_uploads
                    self.id_to_tot_articles[i] += num_uploads
                    mxT = max(mxT, t)
        self.T = mxT
    
    def __str__(self):
        return 'MarkovChainDataset'

    def get_dict_state(self):
        return {'min_lambas': self.min_lambdas, 'max_lambdas': self.max_lambdas, 'seed':self.seed, 'change_state_min_prob': self.change_state_min_prob,
                'change_state_max_prob': self.change_state_max_prob, 'm':self.m, 'T':self.T, 'lambdas': self.lambdas, 'change_probs': self.changeStateP, 'name': self.__str__()}