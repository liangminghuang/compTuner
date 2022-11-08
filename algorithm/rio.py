import os, sys
import time
import random
from .executor import LOG_FILE, write_log

random.seed(123)

opt_seq = [0, 1]


class RIO:
    def __init__(self, get_objective_score, options, iteration=60):
        self.get_objective_score = get_objective_score
        self.iteration = iteration
        self.options = options

    def run(self):
        training_indep = []
        indep = []
        dep = []
        ts = []
        b = time.time()
        while len(training_indep) < self.iteration:
            x = random.randint(0, 2 ** self.options)
            if x not in training_indep:
                training_indep.append(x)
                comb = bin(x).replace('0b', '')
                comb = '0' * (self.options - len(comb)) + comb
                conf = []
                for k, s in enumerate(comb):
                    if s == '1':
                        conf.append(1)
                    else:
                        conf.append(0)
                indep.append(conf)
                dep.append(self.get_objective_score(conf))
                ts.append(time.time() - b)
        print('time:' + str(time.time() - b))
        objectives = [[x, dep[i]] for i, x in enumerate(indep)]
        ss = '{}: best {}'.format(str(round(ts[-1])), str([x[1] for x in objectives]))
        write_log(ss, LOG_FILE)
        return [x[1] for x in objectives], ts

