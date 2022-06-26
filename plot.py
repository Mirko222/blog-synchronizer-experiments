from matplotlib import pyplot as plt
import scipy.special as scsp
from collections import OrderedDict
import os, json
import numpy as np
import gzip
import math
from scipy.optimize import curve_fit

def truncate(x, decimal_places):
    return math.trunc(x * (10**decimal_places)) / (10**decimal_places)

def main(algo_names=['RoundRobin', 'GreedyArticles', 'ExpectedGreedyArticles', 'GreedyPotential', 'ExpectedGreedyPotential', 
                        'OptMemoryless', 'ExactDerandomization', 'EDFDerandomization', 'RandomSchedule'], 
        res_dirs=['results/'], agg='sum', x_axis='rounds', max_x=None, title=None):
    aliases = {'ExactDerandomization': 'RecDerandomization'}
    data = OrderedDict()
    for algo in algo_names:
        data[algo] = []
        for dir in res_dirs:
            for path in sorted(os.listdir(dir)):
                with open(dir+'/'+path, 'r') as f:
                    dt = json.load(f)
                    algo_key = 'algorithm:' if 'algorithm:' in dt else 'algorithm'
                    if algo_key not in dt:
                        print(f'algorithm not found in {dir}/{path}')
                    elif dt[algo_key] == algo:
                        x = dt[x_axis]
                        if max_x is not None and x > max_x:
                            continue
                        if agg == 'sum':
                            y = dt['cost']
                        elif agg == 'avg_m':
                            y = dt['cost']/dt['m']
                        elif agg == 'avg_rounds':
                            y = dt['cost']/dt['rounds']

                        data[algo].append((x, y))

    #plt.style.use('plot_style.txt')
    #fig = plt.figure(figsize=(8,8)) #dpi
    #print(mpl.font_manager.fontManager.ttflist)

    plt.rcParams["font.family"] = "sans-serif" #"STIXGeneral"
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #linestyles = ['--', '-.', '-', '--', '-.', '-']
    linestyles = ['-']
    idx = 0
    for algo in algo_names:
        if len(data[algo]) > 0:
            data[algo] = sorted(set(data[algo]))
            tmp = list(zip(*data[algo]))
            algo_name = algo
            if algo in aliases:
                algo_name = aliases[algo]
            plt.plot(tmp[0], tmp[1], label=algo_name, linewidth=2, linestyle=linestyles[idx%len(linestyles)], marker=None)
            print(algo, tmp[1][-1])
            idx += 1
    plt.title(title)
    ax.ticklabel_format(useMathText=True)
    ax.set_ylabel('cost')
    ax.set_xlabel('rounds')
    ax.legend(frameon=False)
    #plt.grid(True, color='k')
    fig.set_size_inches(7, 6)
    plt.savefig(f'plots/{"_".join(res_dirs).replace("/", "_")}.pdf', bbox_inches='tight')
    plt.show()


def theoreticalCompetitiveRatioPlot(title="theoretical competitive ratios", m=100, T=1000):
    plt.rcParams["font.family"] = "sans-serif" #"STIXGeneral"
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    x = np.arange(start=1, stop=T+1, step=1)
    memoryless = (2 * (x + m - 1) * m) / (x * m + x + m - 1)
    exact_derand = (2 * (x + 2*m + 1) * (x + m - 1)) / (x * x)
    edf_derand = (2 * (x + 2*m + 3) * (x + m - 1)) / (x * x)

    plt.plot(x, memoryless, label="opt memoryless", linewidth=2)
    plt.plot(x, exact_derand, label="rec. derandomization", linewidth=2)
    plt.plot(x, edf_derand, label="EDF derandomization", linewidth=2)

    plt.title(title)
    ax.ticklabel_format(useMathText=True)
    ax.set_ylabel('competitive ratio')
    ax.set_xlabel('T')
    ax.legend(frameon=False)
    #plt.grid(True, color='k')
    fig.set_size_inches(7, 6)
    plt.savefig(f'plots/{title}.pdf', bbox_inches='tight')
    plt.show()

def powerLaw(x, a, k):
    '''power law distribution'''
    return a * (x**k)

def interarrivalTimesHistogram(path, title="", idx=0, max_val=500, save_path='interarrival_times'):
    plt.rcParams["font.family"] = "sans-serif" #"STIXGeneral"
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    lines = []
    with gzip.open(path, 'rt') as f:
        lines = f.readlines()
    lines = sorted(lines, key=lambda v: len(v.split()), reverse=True)

    timestamps = sorted(list(map(int, lines[idx].split()[1:])))
    interarrivals = []
    for i in range(len(timestamps)-1):
        interarrivals.append(timestamps[i+1] - timestamps[i])
    interarrivals = sorted(interarrivals)
    counts, bins, _ = plt.hist(interarrivals, bins=max_val, range=(1, max_val), histtype='stepfilled', density=True) #stepfilled
    #set(len(interarrivals))
    sp = bins[1] - bins[0]
    bins = bins[:-1] #- sp/2
    #bins = bins[:-1]
    param_opt, _ = curve_fit(powerLaw, bins, counts, maxfev=3000) #fit curve
    x_curve = np.linspace(1, max_val, 1000)
    y_curve = powerLaw(x_curve, *param_opt)
    param_opt[0] 
    lab = f'{truncate(param_opt[0], 3)} x^{truncate(param_opt[1], 3)}'
    plt.plot(x_curve, y_curve, label=lab, linestyle='--')
    
    plt.title(title)
    ax.ticklabel_format(useMathText=True)
    ax.set_ylabel('probability')
    ax.set_xlabel('interarrival times')
    ax.legend(frameon=False)
    fig.set_size_inches(7, 6)
    plt.savefig(f'plots/{save_path}.pdf', bbox_inches='tight')
    plt.show()

def convergenceEstimationGraph(algo_names=['RoundRobin', 'GreedyArticles', 'ExpectedGreedyArticles', 'GreedyPotential', 
                    'ExpectedGreedyPotential', 'OptMemoryless', 'EDFDerandomization', 'RandomSchedule'], 
                    res_dirs=['results/'], title='', error_func='l1', agg_func = 'sum', single_idx = None, save_path='est_conv', y_label='error'):
    aliases = {'ExactDerandomization': 'RecDerandomization'}
    data = OrderedDict()
    min_real = 9999999
    max_real = -1
    for algo in algo_names:
        data[algo] = []
        real_val_key = 'lambdas'
        estimation_key = 'lambda_estimates'
        round_key = 'rounds'
        for dir in res_dirs:
            for path in sorted(os.listdir(dir)):
                with open(dir+'/'+path, 'r') as f:
                    dt = json.load(f)
                    if 'name' not in dt['dataset']:
                        dt['dataset']['name'] = 'MarkovChainDataset'
                    algo_key = 'algorithm:' if 'algorithm:' in dt else 'algorithm'
                    if algo_key not in dt:
                        print(f'algorithm not found in {dir}/{path}')
                    elif dt[algo_key] == algo:
                        assert(estimation_key in dt and round_key in dt and real_val_key in dt['dataset'])
                        if dt[round_key] > dt['T']:
                            continue
                        error = 0
                        blogs = range(len(dt[estimation_key])) if single_idx is None else [single_idx]
                        for i in blogs:
                            est = dt[estimation_key][i]
                            if dt['dataset']['name'] == 'PoissonDataset':
                                real = dt['dataset'][real_val_key][i]
                            elif dt['dataset']['name'] == 'RenewalProcessDataset' and dt['dataset']['inter_arrival'] == 'zeta':
                                real = dt['dataset'][real_val_key][i]
                                if real > 2:
                                    real = scsp.zeta(real) / scsp.zeta(real - 1)
                                else:
                                    real = 0.1
                            elif dt['dataset']['name'] == 'MarkovChainDataset':
                                pa = dt['dataset']['change_probs'][1][i] / (dt['dataset']['change_probs'][1][i] + dt['dataset']['change_probs'][0][i])
                                real = pa * dt['dataset']['lambdas'][0][i] + (1 - pa) * dt['dataset']['lambdas'][1][i]
                            min_real = min(real, min_real)
                            max_real = max(real, max_real)
                            if error_func == 'l1':
                                error += abs(est - real)
                            elif error_func == 'l2':
                                error += (est - real) * (est - real)
                            elif error_func == 'lmax':
                                error = max(error, abs(est - real))
                            elif error_func == 'l1norm':
                                error += (abs(est - real) / real)
                        if agg_func == 'avg' and single_idx is None:
                            error /= dt['m']
                        data[algo].append((dt[round_key], error))
    
    print('min real rate:', min_real)
    print('max real rate:', max_real)
    plt.rcParams["font.family"] = "sans-serif" #"STIXGeneral"
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #linestyles = ['--', '-.', '-', '--', '-.', '-']
    linestyles = ['-']
    idx = 0
    for algo in algo_names:
        if len(data[algo]) > 0:
            data[algo] = sorted(set(data[algo]))
            tmp = list(zip(*data[algo]))
            algo_name = algo
            if algo in aliases:
                algo_name = aliases[algo]
            plt.plot(tmp[0], tmp[1], label=algo_name, linewidth=2, linestyle=linestyles[idx%len(linestyles)], marker=None)
            print(algo, tmp[1][-1])
            idx += 1
    plt.title(title)
    ax.ticklabel_format(useMathText=True)
    ax.set_ylabel(y_label)
    ax.set_xlabel('rounds')
    ax.legend(frameon=False)
    #plt.grid(True, color='k')
    fig.set_size_inches(7, 6)
    plt.savefig(f'plots/{save_path}.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main(res_dirs=['results/200k', 'results/200k_random', 'results/200k_derand'], agg='sum', title="k=1")
    main(res_dirs=['results/200k_k2'], agg='sum', title="k=2")
    main(res_dirs=['results/200k_k3'], agg='sum', title="k=3")
    main(res_dirs=['results/500k_31files_15minarr'], agg='sum')
    main(res_dirs=['results/250k_31files_30minarr'], agg='sum')
    main(res_dirs=['results/2.5M_31files_50minarr'], agg='sum')

    interarrivalTimesHistogram('../blogs/reddit_2020-01-01.gz', idx=1, title='')

    main(res_dirs=['results/synthetic/poisson_m96_T250000_power2_v2'], agg='sum', title='')
    convergenceEstimationGraph(res_dirs=['results/synthetic/poisson_m96_T250000_power2_v2'], error_func='l1norm', agg_func='avg', single_idx = None, save_path='poisson_estimation_conv')

    main(res_dirs=['results/synthetic/markovchain_08-001_1-005_0001-001_m100_T250000_init1'], agg='sum', title='')
    convergenceEstimationGraph(res_dirs=['results/synthetic/markovchain_08-001_1-005_0001-001_m100_T250000_init1'], error_func='l1norm', agg_func='avg', single_idx = None, title='', save_path='markov_chain_est_conv')
    
    main(res_dirs=['results/synthetic/powerlaw_m100_T2000000_2.01_3'], agg='sum', title='')
    convergenceEstimationGraph(res_dirs=['results/synthetic/powerlaw_m100_T2000000_2.01_3'], error_func='l1norm', agg_func='avg', single_idx = None, title='', save_path='powerlaw_est_conv_2.01')
    main(res_dirs=['results/synthetic/powerlaw_m100_T2000000_1.01_3'], agg='sum', title='')
    convergenceEstimationGraph(res_dirs=['results/synthetic/powerlaw_m100_T2000000_1.01_3'], error_func='l1norm', agg_func='avg', single_idx = None, title='', save_path='powerlaw_est_conv_1.01')
    