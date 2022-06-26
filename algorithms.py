from collections import OrderedDict

from dataset import *
from util import *
import random, math, tqdm, json, time

def lower_bound(dataset):
    cost = 0
    for (_, blogs) in dataset.sparse_matrix.items():
        vals = sorted(list(blogs.values()), reverse=True)
        cost += sum(vals[i]*(i+1) for i in range(len(vals)))
    return cost

class Algorithm():
    '''general algorithm'''
    def __init__(self, dataset, compute_potential=False, disable_tqdm=True, use_ordered_dict = False, 
                    dump_frequency=None, dump_directory=None, k=1, dump_lambda_estimate=False):
        self.dataset = dataset
        self.k = k
        self.dump_lambda_estimate = dump_lambda_estimate

        if use_ordered_dict:
            self.num_articles = OrderedDict()
        else:
            self.num_articles = {}
        
        if use_ordered_dict:
            self.potential = OrderedDict()
        else:
            self.potential = {}
        self.sum_of_nonempty_potential = 0
        self.max_potential_id = -1

        self.round = 0
        self.cost = 0
        self.T = dataset.T
        self.m = dataset.m
        
        self.dump_cnt = 0
        rounds_to_dump = dump_frequency
        
        self.tot_articles = 0
        self.last_downloaded = [-1] * self.m
        self.sum_articles_so_far_per_blog = [0] * self.m
        self.downloaded_articles_so_far_per_blog = [0] * self.m

        non_empty_rounds = sorted(list(dataset.sparse_matrix.keys()))
        self.round = non_empty_rounds[0]
        non_empty_idx = 1
        
        #repeat until all blogs are empty and T has been reached
        with tqdm.tqdm(total=self.T, disable=disable_tqdm) as pbar:
            while self.round <= dataset.T or self.tot_articles > 0:
                if self.round in dataset.sparse_matrix:
                    for (blog_id, n_articles) in dataset.sparse_matrix[self.round].items():
                        if blog_id not in self.num_articles:
                            self.num_articles[blog_id] = 0
                        self.num_articles[blog_id] += n_articles
                        self.tot_articles += n_articles
                        self.sum_articles_so_far_per_blog[blog_id] += n_articles
                
                #compute potentials only if needed
                if compute_potential:
                    max_potential = -1
                    self.sum_of_nonempty_potential = 0
                    self.max_potential_id = 0
                    for (id, num_art) in self.num_articles.items():
                        if id not in self.potential:
                            self.potential[id] = 0
                        self.potential[id] += num_art
                        pot = self.potential[id]
                        self.sum_of_nonempty_potential += pot
                        if pot > max_potential or (pot == max_potential and id < self.max_potential_id):
                            max_potential = pot
                            self.max_potential_id = id

                #update cost and download k blogs
                self.cost += self.tot_articles
                for _ in range(self.k):
                    download = self.choose_download()
                    self.last_downloaded[download] = self.round
                    self.downloaded_articles_so_far_per_blog[download] = self.sum_articles_so_far_per_blog[download]

                    if download in self.num_articles:
                        self.tot_articles -= self.num_articles[download]
                        del self.num_articles[download]
                        if compute_potential:
                            self.sum_of_nonempty_potential -= self.potential[download]
                            self.potential.pop(download, None)
                        #max potential is invalid for k>1
                        self.max_potential_id = -1

                assert(self.tot_articles >= 0)
                
                #go to next round
                while non_empty_idx < len(non_empty_rounds) and non_empty_rounds[non_empty_idx] <= self.round:
                    non_empty_idx += 1
                if self.tot_articles > 0 or non_empty_idx >= len(non_empty_rounds):
                    self.round += 1
                    pbar.update(1)
                    if dump_frequency is not None:
                        rounds_to_dump -= 1
                else:
                    pbar.update(non_empty_rounds[non_empty_idx] - self.round)
                    if dump_frequency is not None:
                        rounds_to_dump -= non_empty_rounds[non_empty_idx] - self.round
                    self.round = non_empty_rounds[non_empty_idx]

                if dump_frequency is not None and dump_directory is not None and rounds_to_dump <= 0:
                    self.save_results(dump_directory)
                    rounds_to_dump = dump_frequency

            if dump_directory is not None:
                self.save_results(dump_directory)
    
    def get_dict_state(self):
        d = {'T': self.T, 'm':self.m, 'rounds': self.round, 'cost': self.cost, 'algorithm': self.__str__(),
             'k':self.k, 'dataset': self.dataset.get_dict_state()}
        if self.dump_lambda_estimate:
            l_est = [self.downloaded_articles_so_far_per_blog[i]/(self.round+1) for i in range(self.m)]
            d['lambda_estimates'] = l_est
        return d

    def save_results(self, path):
        with open(path+str(self.dump_cnt)+'-'+str(time.time())+'.json', 'w') as f:
            json.dump(self.get_dict_state(), f, indent = 6)
        self.dump_cnt += 1

    def choose_download(self):
        pass

class RoundRobin(Algorithm):
    def __init__(self, dataset, check_nonempty=False, dump_frequency=None, 
                    dump_directory=None, k=1, dump_lambda_estimate=False):
        self.idx = 0
        self.check_nonempty = check_nonempty
        super().__init__(dataset, dump_frequency=dump_frequency, dump_directory=dump_directory, 
                            k=k, dump_lambda_estimate=dump_lambda_estimate)
    
    def choose_download(self):
        if not self.check_nonempty:
            download = self.idx
            self.idx = (self.idx+1)%self.m
            return download
        assert(self.check_nonempty)
        mn = self.m-1
        download = self.m+1
        for id in self.num_articles:
            if id < mn:
                mn = id
            if id >= self.idx and id < download:
                download = id
        if download >= self.m:
            download = mn
        self.idx = (download+1)%self.m
        return download        
    
    def __str__(self):
        return 'RoundRobin'

    def get_dict_state(self):
        d = super().get_dict_state()
        d['check_nonempty'] = self.check_nonempty
        return d

class GreedyArticles(Algorithm):
    def __init__(self, dataset, dump_frequency=None, dump_directory=None, k=1):
        super().__init__(dataset, dump_frequency=dump_frequency, dump_directory=dump_directory, k=k)
    
    def __str__(self):
        return 'GreedyArticles'

    def choose_download(self):
        download = 0
        best_num = 0
        for (id, num) in self.num_articles.items():
            if num > best_num or (num == best_num and id < download):
                download = id
                best_num = num 
        return download

class ExpectedGreedyArticles(Algorithm):
    def __init__(self, dataset, dump_frequency=None, dump_directory=None, k=1, pull_lambda_estimate=False, dump_lambda_estimate=False):
        assert(k==1) #doesn't handle k>1
        self.act_art = [0] * dataset.m
        self.pull_lambda_estimate = pull_lambda_estimate
        super().__init__(dataset, dump_frequency=dump_frequency, dump_directory=dump_directory, k=k, dump_lambda_estimate=dump_lambda_estimate)
    
    def __str__(self):
        return 'ExpectedGreedyArticles'

    def choose_download(self):
        best_id, best_val = 0, 0
        for id in range(self.m):
            if not self.pull_lambda_estimate:
                l = self.sum_articles_so_far_per_blog[id] / (self.round+1)
            else:
                l = max(1, self.downloaded_articles_so_far_per_blog[id]) / max(1, self.last_downloaded[id])
            self.act_art[id] += l
            if self.act_art[id] > best_val:
                best_val = self.act_art[id]
                best_id = id
        self.act_art[best_id] = 0
        return best_id

    def get_dict_state(self):
        d = super().get_dict_state()
        d['pull_lambda_estimate'] = self.pull_lambda_estimate
        return d

class GreedyPotential(Algorithm):
    def __init__(self, dataset, dump_frequency=None, dump_directory=None, k=1):
        super().__init__(dataset, compute_potential=True, dump_frequency=dump_frequency, dump_directory=dump_directory, k=k)
    
    def __str__(self):
        return 'GreedyPotential'

    def choose_download(self):
        if self.max_potential_id >= 0: #maximum potential already computed
            return self.max_potential_id
        #no speed-up for k>1
        assert(self.k > 1)
        bid, bp = self.m-1, -1
        for (id, p) in self.potential.items():
            if p > bp or (p == bp and id < bid):
                bp = p
                bid = id
        return bid

class RandomizedGreedyPotential(Algorithm):
    def __init__(self, dataset, seed=42, k=1):
        self.seed = seed
        random.seed(seed)
        super().__init__(dataset, compute_potential=True, k=k)
    
    def __str__(self):
        return 'RandomizedGreedyPotential'

    def choose_download(self):
        r = random.random() * self.sum_of_nonempty_potential
        for (id, pot) in self.potential.items():
            if r <= pot:
                return id
            r -= pot
        assert(False)
    
    def get_dict_state(self):
        d = super().get_dict_state()
        d['seed'] = self.seed
        return d

class ExpectedGreedyPotential(Algorithm):
    def __init__(self, dataset, dump_frequency=None, dump_directory=None, k=1, pull_lambda_estimate=False, dump_lambda_estimate=False):
        assert(k==1) #doesn't handle k>1
        self.act_pot = [0] * dataset.m
        self.pull_lambda_estimate = pull_lambda_estimate
        super().__init__(dataset, dump_frequency=dump_frequency, dump_directory=dump_directory, k=k, dump_lambda_estimate=dump_lambda_estimate)
    
    def __str__(self):
        return 'ExpectedGreedyPotential'

    def choose_download(self):
        best_id, best_val = 0, 0
        for id in range(self.m):
            t = self.round - self.last_downloaded[id]
            if not self.pull_lambda_estimate:
                l = self.sum_articles_so_far_per_blog[id] / (self.round+1)
            else:
                l = max(1, self.downloaded_articles_so_far_per_blog[id]) / max(1, self.last_downloaded[id])
            self.act_pot[id] += t * l

            if self.act_pot[id] > best_val:
                best_val = self.act_pot[id]
                best_id = id
        
        self.act_pot[best_id] = 0
        return best_id
    
    def get_dict_state(self):
        d = super().get_dict_state()
        d['pull_lambda_estimate'] = self.pull_lambda_estimate
        return d

class OptMemoryless(Algorithm):
    def __init__(self, dataset, check_nonempty=True, seed=23, dump_frequency=None, dump_directory=None, 
                    k=1, pull_lambda_estimate = False, dump_lambda_estimate=False):
        assert((not pull_lambda_estimate) or k==1) #doesn't handle k>1 and pull_lambda_estimate
        self.seed = seed
        self.check_nonempty = check_nonempty
        self.pull_lambda_estimate = pull_lambda_estimate
        random.seed(seed)
        super().__init__(dataset, dump_frequency=dump_frequency, dump_directory=dump_directory, 
                            k=k, dump_lambda_estimate=dump_lambda_estimate)
    
    def __str__(self):
        return 'OptMemoryless'

    def choose_download(self):
        ids = sorted(self.num_articles.keys()) if self.check_nonempty else list(range(self.m))
        if not self.pull_lambda_estimate:
            sqrt_lambdas = [math.sqrt(self.sum_articles_so_far_per_blog[id]/(self.round+1)) for id in ids]
        else:
            sqrt_lambdas = [math.sqrt(max(1, self.downloaded_articles_so_far_per_blog[id]) / 
                                max(1, self.last_downloaded[id])) for id in ids]
        r = random.random() * sum(sqrt_lambdas)
        for i in range(len(ids)):
            if r < sqrt_lambdas[i]:
                return ids[i]
            r -= sqrt_lambdas[i]
        return 0 if len(ids) == 0 else ids[-1]

    def get_dict_state(self):
        d = super().get_dict_state()
        d['seed'] = self.seed
        d['check_nonempty'] = self.check_nonempty
        d['pull_lambda_estimate'] = self.pull_lambda_estimate
        return d

class EDFDerandomization(Algorithm):
    def __init__(self, dataset, check_nonempty=True, dump_frequency=None, dump_directory=None, 
                    k=1, pull_lambda_estimate=False, dump_lambda_estimate=False):
        assert((not pull_lambda_estimate) or k==1) #doesn't handle k>1 and pull_lambda_estimate
        self.delta = [0] * dataset.m
        self.lambdas = [0] * dataset.m
        self.check_nonempty = check_nonempty
        self.pull_lambda_estimate = pull_lambda_estimate
        super().__init__(dataset, use_ordered_dict=True, dump_frequency=dump_frequency, 
                    dump_directory=dump_directory, k=k, dump_lambda_estimate=dump_lambda_estimate)

    def __str__(self):
        return 'EDFDerandomization'

    def choose_download(self):
        sum_of_lambdas = 0
        ids = self.num_articles if self.check_nonempty else range(self.m)

        for id in ids:
            if not self.pull_lambda_estimate:
                self.lambdas[id] = math.sqrt(self.sum_articles_so_far_per_blog[id] / (self.round+1))
            else:
                self.lambdas[id] = math.sqrt(max(1, self.downloaded_articles_so_far_per_blog[id]) / 
                                                max(1, self.last_downloaded[id]))
            sum_of_lambdas += self.lambdas[id]
        
        backup, backupval = 0, -1
        bestId, bestVal = -1, 1
        for id in ids:
            self.delta[id] += (self.lambdas[id] / sum_of_lambdas)
            if self.delta[id] > backupval:
                backupval = self.delta[id]
                backup = id
            if self.delta[id] > 0:
                d = (1 - self.delta[id]) / (self.lambdas[id] / sum_of_lambdas)
                if bestId == -1 or d < bestVal:
                    bestId = id
                    bestVal = d

        if bestId >= 0:
            self.delta[bestId] -= 1
            return bestId
        return backup
    
    def get_dict_state(self):
        d = super().get_dict_state()
        d['min_lambda'] = min(self.lambdas)
        d['max_lambda'] = max(self.lambdas)
        d['check_nonempty'] = self.check_nonempty
        d['pull_lambda_estimate'] = self.pull_lambda_estimate
        return d        

class ExactDerandomization(Algorithm):
    def __init__(self, dataset, check_nonempty=True, dump_frequency=None, dump_directory=None, k=1):
        self.ri = dataset.getDerandomizationRi()
        self.cycle = self.build_cycle(max(self.ri), OrderedDict([(i,self.ri[i]) for i in range(dataset.m)]))

        last = [0] * dataset.m
        for (round, id) in enumerate(self.cycle):
            assert(round - last[id] <= 2**self.ri[id])
            last[id] = round

        #build id_to_indices to use binary search later
        self.id_to_indices = {}
        for i in range(len(self.cycle)):
            if self.cycle[i] not in self.id_to_indices:
                self.id_to_indices[self.cycle[i]] = []
            self.id_to_indices[self.cycle[i]].append(i)

        self.cnt = 0
        self.check_nonempty = check_nonempty

        super().__init__(dataset, use_ordered_dict=True, dump_frequency=dump_frequency, dump_directory=dump_directory, k=k)

    def build_cycle(self, rho, ris):
        if rho == 0:
            assert(len(ris.keys()) == 1)
            return [list(ris.keys())[0]]
        
        #find smallest 2^-r_i blogs
        min_ids = [id for (id, r) in ris.items() if r == rho]
        if len(min_ids) == 1:
            ris[min_ids[0]] -= 1
            return self.build_cycle(rho-1, ris)
        if len(min_ids)%2 == 1:
            ris[min_ids[-1]] -= 1
            min_ids = min_ids[:-1]
        
        #make pair of blogs with smallest 2^-r_i
        toggle = {}
        adj = {}
        for i in range(0, len(min_ids), 2):
            toggle[min_ids[i]] = True
            adj[min_ids[i]] = min_ids[i+1]
            ris[min_ids[i]] -= 1
            del ris[min_ids[i+1]]
        
        #build cycle recursively and duplicate adjusting the pairs
        cycle = self.build_cycle(rho-1, ris)
        cycle.extend(cycle)
        for i in range(len(cycle)):
            if cycle[i] in toggle:
                toggle[cycle[i]] = not toggle[cycle[i]]
                if toggle[cycle[i]]:
                    cycle[i] = adj[cycle[i]]
    
        return cycle

    def choose_download(self):
        if not self.check_nonempty:
            download = self.cycle[self.cnt]
            self.cnt = (self.cnt + 1) % len(self.cycle)
            return download
        backup = -1
        best_index = -1
        #iterate non-empty blogs and take the successive in the cycle
        for id in self.num_articles:
            idx = upper_bound_non_strict(self.id_to_indices[id], self.cnt)
            if idx < len(self.id_to_indices[id]) and (best_index == -1 or self.id_to_indices[id][idx] < best_index):
                best_index = self.id_to_indices[id][idx]
            if backup == -1 or self.id_to_indices[id][0] < backup:
                backup = self.id_to_indices[id][0]
        if best_index == -1:
            best_index = backup
        if best_index == -1:
            return 0
        self.cnt = (best_index + 1) % len(self.cycle)
        return self.cycle[best_index]
    
    def __str__(self):
        return 'ExactDerandomization'

    def get_dict_state(self):
        d = super().get_dict_state()
        d['check_nonempty'] = self.check_nonempty
        d['cycle_length'] = len(self.cycle)
        return d        

class RandomSchedule(Algorithm):
    def __init__(self, dataset, seed=42, check_nonempty=True, k=1, dump_frequency=None, dump_directory=None, dump_lambda_estimate=False):
        self.seed = seed
        random.seed(seed)
        self.check_nonempty = check_nonempty
        super().__init__(dataset, k=k, dump_lambda_estimate=dump_lambda_estimate, dump_frequency=dump_frequency, dump_directory=dump_directory)
    
    def __str__(self):
        return 'RandomSchedule'

    def choose_download(self):
        if self.check_nonempty:
            return random.choice(sorted(self.num_articles.keys()))
        return random.randint(0, self.m-1)

    def get_dict_state(self):
        d = super().get_dict_state()
        d['seed'] = self.seed
        return d