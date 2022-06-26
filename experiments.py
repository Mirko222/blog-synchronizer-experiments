from algorithms import *
from dataset import *

if __name__ == '__main__':
    # ====== choose output directory and dump frequency
    results_directory = ''
    dump_freq = 1000
    k = 1
    #results_directory = None
    #dump_freq = None
    
    # ======= choose dataset
    min_ts = 1577833586
    lambdas = [2/(2**i) for i in range(1, 13)] * 8
    #ds = PoissonDataset(lambdas, 250000, seed=55, min_l=0.5, max_l=2.5)
    #ds = RenewalProcessDataset(100, 5000, seed=55, inter_arrival='zeta', min_l=2.01, max_l=3, lambda_sample='uniform') #1.01, 3
    #ds = MarkovChainDataset(250000, 100, min_lambdas=[0.8, 0.01], max_lambdas=[1, 0.05], change_state_min_prob=[0.001, 0.001], change_state_max_prob=[0.01, 0.01], init_state=1, seed=58)
    ds = Dataset('../blogs/reddit_2020-01-01.gz', discr_step=1, min_number_of_arrivals = 1, max_timestamp=min_ts+200000)
    
    print('lower bound cost:', lower_bound(ds))

    # ====== choose algorithms to run

    #------- push-based
    rr = RoundRobin(ds, check_nonempty=True, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('round robin cost: ', rr.cost)
    
    gp = GreedyPotential(ds, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('greedy potential cost: ', gp.cost)

    edfDeran = EDFDerandomization(ds, check_nonempty=True, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('(EDF) derandomized: ', edfDeran.cost)

    greedyA = GreedyArticles(ds, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('greedy articles cost: ', greedyA.cost)
    
    exactDerand = ExactDerandomization(ds, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('rec derandomization cost: ', exactDerand.cost)
    
    optMem = OptMemoryless(ds, seed=23, dump_frequency=dump_freq, dump_directory=results_directory, k=k)
    print('opt memoryless cost: ', optMem.cost)
    
    #randomizedGP = RandomizedGreedyPotential(ds, seed=42)
    #print('randomized greedy potential cost: ', randomizedGP.cost)


    #-------- pull based

    rrEM = RoundRobin(ds, check_nonempty=False, dump_frequency=dump_freq, dump_directory=results_directory, dump_lambda_estimate=True)
    print('round robin cost (pull): ', rrEM.cost)
    
    #can be used, but it uses the final estimates
    #exactDerandEM = ExactDerandomization(ds, check_nonempty=False, dump_frequency=dump_freq, dump_directory=results_directory)
    #print('rec derandomization cost (pull): ', exactDerandEM.cost)

    expectedGP = ExpectedGreedyPotential(ds, dump_frequency=dump_freq, dump_directory=results_directory, pull_lambda_estimate=True, dump_lambda_estimate=True)
    print('expected greedy potential cost: ', expectedGP.cost)

    edfDeranEM = EDFDerandomization(ds, check_nonempty=False, dump_frequency=dump_freq, dump_directory=results_directory, pull_lambda_estimate=True, dump_lambda_estimate=True)
    print('(EDF) derandomized (pull): ', edfDeranEM.cost)

    expectedGA = ExpectedGreedyArticles(ds, dump_frequency=dump_freq, dump_directory=results_directory, pull_lambda_estimate=True, dump_lambda_estimate=True)
    print('expected greedy articles cost: ', expectedGA.cost)

    optMemEM = OptMemoryless(ds, seed=42, check_nonempty=False, dump_frequency=dump_freq, dump_directory=results_directory, pull_lambda_estimate=True, dump_lambda_estimate=True)
    print('opt memoryless cost (pull): ', optMemEM.cost)

    randomSched = RandomSchedule(ds, seed=42, check_nonempty=False, dump_frequency=dump_freq, dump_directory=results_directory, dump_lambda_estimate=True)
    print('random schedule cost (pull): ', randomSched.cost)
    
