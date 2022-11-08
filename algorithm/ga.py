import argparse
import os, sys
import random
import numpy as np
import time
random.seed(123)
initial_set = 4
begin2end = 3
iters = 100

from .executor import Executor, LOG_DIR

class GA:
    def __init__(self, options, get_objective_score):
        self.options  = options
        self.get_objective_score = get_objective_score
        geneinfo = []
        for i in range(initial_set):
            x = random.randint(0, 2 ** len(self.options))
            geneinfo.append(self.generate_conf(x))
        fitness = []
        self.begin = time.time()
        self.dep = []
        self.times = []
        for x in geneinfo:
            tmp = self.get_objective_score(x)
            fitness.append(-1.0 / tmp)
            
        self.pop = [(x, fitness[i]) for i, x in enumerate(geneinfo)]
        self.pop = sorted(self.pop, key=lambda x:x[1])
        self.best = self.selectBest(self.pop)
        self.dep.append(1.0/self.best[1])
        self.times.append(time.time() - self.begin)

    def generate_conf(self, x):
        comb = bin(x).replace('0b', '')
        comb = '0' * (len(self.options) - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def selectBest(self, pop):
        return pop[0]
        
    def selection(self, inds, k):
        s_inds = sorted(inds, key=lambda x:x[1])
        return s_inds[:int(k)]

    def crossoperate(self, offspring):
        dim = len(self.options)
        geninfo1 = offspring[0][0]
        geninfo2 = offspring[1][0]
        pos = random.randrange(1, dim)

        newoff = []
        for i in range(dim):
            if i>=pos:
                newoff.append(geninfo2[i])
            else:
                newoff.append(geninfo1[i])
        return newoff

    def mutation(self, crossoff):
        dim = len(self.options)
        pos = random.randrange(1, dim)
        crossoff[pos] = 1 - crossoff[pos]
        return crossoff

    def GA_main(self):
        for g in range(iters):
            selectpop = self.selection(self.pop, 0.5 * initial_set)
            nextoff = []
            while len(nextoff) != initial_set:
                offspring = [random.choice(selectpop) for i in range(2)]
                crossoff = self.crossoperate(offspring)
                muteoff = self.mutation(crossoff)
                fit_muteoff = self.get_objective_score(muteoff)
                nextoff.append((muteoff, -1.0 / fit_muteoff))
            self.pop = nextoff       
            self.pop = sorted(self.pop, key=lambda x:x[1])
            self.best = self.selectBest(self.pop)
            self.times.append(time.time() - self.begin)
            self.dep.append(1.0/self.best[1])

        return self.dep, self.times
