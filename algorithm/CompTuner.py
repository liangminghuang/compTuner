import time, math
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from .executor import LOG_FILE, write_log

opt_seq = [0, 1]

class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent  # fix values of impactful opts

    def to_next(self, opt_ids, l):
        """
        Flip selected less-impactful opt, then fix impactful optimization
        """
        ans = [0] * l
        for f in opt_ids:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1]
        return ans

class compTuner:
    def __init__(self, dim, iteration, c1, c2, w, get_objective_score, random):
        """
        :param dim: number of compiler flags
        :param iteration: number of iteration
        :param c1: parameter of pso process
        :param c2: parameter of pso process
        :param w: parameter of pso process
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        """
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.iteration = iteration
        self.dim = dim
        self.V = []
        self.pbest = []
        self.gbest = []
        self.p_fit = []
        self.fit = 0
        self.get_objective_score = get_objective_score
        self.random = random

    def generate_random_conf(self, x):
        """
        Generation 0-1 mapping for disable-enable options
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds:Sequences' speedup for EI
        :param eta:global best speedup
        :return:the EI for a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model:RandomForest Model
        :param now_best:global best speedup
        :param wait_for_train:Sequences Set
        :return:the Sequences' EI
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]

    def runtime_predict(self, model, wait_for_train):
        """
        :param model:model:RandomForest Model
        :param wait_for_train:Sequences Set
        :return: the speedup of Sequences Set
        """
        preds_result = []
        estimators = model.estimators_
        t = 1
        for e in estimators:
            tmp = e.predict(np.array(wait_for_train))
            if t == 1:
                for i in range(len(tmp)):
                    preds_result.append(tmp)
                t = t + 1
            else:
                for i in range(len(tmp)):
                    preds_result[i] = preds_result[i] + tmp
                t = t + 1
            print(preds_result)
        for i in range(len(preds_result)):
            preds_result[i] = preds_result[i] / (t - 1)
        a = []

        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], preds_result[0][i]]
            a.append(x)
        return a

    def getDistance(self, seq1, seq2):
        """
        :param seq1:
        :param seq2:
        :return: Getting the diversity of two sequences
        """
        t1 = np.array(seq1)
        t2 = np.array(seq2)
        s1_norm = np.linalg.norm(t1)
        s2_norm = np.linalg.norm(t2)
        cos = np.dot(t1, t2) / (s1_norm * s2_norm)
        """
        remove eight flags
        """
        return cos

    def getPrecision(self, model, seq):
        """
        :param model:
        :param seq:
        :return: The precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(seq, k_iter=100086)
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running

    def build_RF_by_BOCA(self):
        """
        Ablation study part 1
        :return:
        """
        inital_indep = []
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, k_iter=0) for indep in inital_indep]
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        while len(inital_indep) < 60:
            neighbors = []
            for i in range(80000):
                x = random.randint(0, 2 ** self.dim)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            estimators = model.estimators_
            global_best = max(inital_dep)
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    inital_dep.append(self.get_objective_score(x[0], k_iter=0))
                    flag = True

        return model, inital_indep, inital_dep

    def build_RF_by_CompTuner(self):
        """
        :return: model, initial_indep, initial_dep
        """
        inital_indep = []
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, k_iter=0) for indep in inital_indep]
        all_acc = []
        model = RandomForestRegressor(random_state=self.random)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 0
        while rec_size < 50:
            model = RandomForestRegressor(random_state=self.random)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(80000):
                x = random.randint(0, 2 ** self.dim)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1

            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1

            if rec_size > 50 and np.mean(all_acc) < 0.04:
                break
        return model, inital_indep, inital_dep

    def selectByDistribution(self, merged_predicted_objectives):
        """
        :param merged_predicted_objectives: sorts sequences by EI value
        :return: selected sequence index
        """
        fitness = np.zeros(len(merged_predicted_objectives),)
        probabilityTotal = np.zeros(len(fitness))
        rec = 0.0000125
        proTmp = 0.0
        for i in range(len(fitness)):
            fitness[i] = random.uniform(0, (i+1) * rec)
            proTmp += fitness[i]
            probabilityTotal[i] = proTmp
        randomNumber = np.random.rand()
        result = 0
        for i in range(1, len(fitness)):
            if randomNumber < fitness[0]:
                result = 0
                break
            elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
                result = i
        return result

    def search_by_impactful(self, model, eta):
        """
        :param model: prediction model
        :param eta: best perfomance of the train set
        :return: new generated sequences
        """
        options = model.feature_importances_
        opt_sort = [[i, x] for i, x in enumerate(options)]
        opt_selected = sorted(opt_sort, key=lambda x: x[1], reverse=True)[:8]
        opt_ids = [x[0] for x in opt_sort]
        neighborhood_iterators = []

        for i in range(2 ** 8):  # search all combinations of impactful optimization
            comb = bin(i).replace('0b', '')
            comb = '0' * (8 - len(comb)) + comb  # fnum-size 0-1 string
            inc = []  # list of tuple: (opt_k's idx, enable/disable)
            for k, s in enumerate(comb):
                if s == '1':
                    inc.append((opt_selected[k][0], 1))
                else:
                    inc.append((opt_selected[k][0], 0))
            neighborhood_iterators.append(get_exchange(inc))

        neighbors = []  # candidate seq
        for i, inc in enumerate(neighborhood_iterators):
            for _ in range(1 + 80000):
                flip_n = random.randint(0, self.dim)
                selected_opt_ids = random.sample(opt_ids, flip_n)
                neighbor_iter = neighborhood_iterators[i].to_next(selected_opt_ids, self.dim)
                neighbors.append(neighbor_iter)
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(neighbors)))
        acq_val_incumbent = self.get_ei(preds, eta)

        return [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]

    def init_v(self, n, d, V_max, V_min):
        """
        :param n: number of particles
        :param d: number of compiler flags
        :return: particle's initial velocity vector
        """
        v = []
        for i in range(n):
            vi = []
            for j in range(d):
                a = random.random() * (V_max - V_min) + V_min
                vi.append(a)
            v.append(vi)
        return v

    def update_v(self, v, x, m, n, pbest, g, w, c1, c2, vmax, vmin):
        """
        :param v: particle's velocity vector
        :param x: particle's position vector
        :param m: number of partical
        :param n: number of compiler flags
        :param pbest: each particle's best position vector
        :param g: all particles' best position vector
        :param w: weight parameter
        :param c1: control parameter
        :param c2: control parameter
        :param vmax: max V
        :param vmin: min V
        :return: each particle's new velocity vector
        """
        for i in range(m):
            a = random.random()
            b = random.random()
            for j in range(n):
                v[i][j] = w * v[i][j] + c1 * a * (pbest[i][j] - x[i][j]) + c2 * b * (g[j] - x[i][j])
                if v[i][j] < vmin:
                    v[i][j] = vmin
                if v[i][j] > vmax:
                    v[i][j] = vmax
        return v

    def run(self):
        begin = time.time()
        """
        build model and get data set
        """
        ts = []
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        time_set_up = 6000
        ts.append(time.time() - begin)
        # model, inital_indep, inital_dep = self.build_RF_by_BOCA() # BOCA build method
        """
        run in impactful method
        """
        # global_best = max(inital_dep)
        # idx = inital_dep.index(global_best)
        # global_best_indep = inital_indep[idx]
        # for iter in range(250):
        #     merged_predicted_objectives = self.search_by_impactful(model, global_best)
        #     flag = False
        #     for x in merged_predicted_objectives:
        #         if flag:
        #             break
        #         if x[0] not in inital_indep:
        #             inital_indep.append(x[0])
        #             speed_up = self.get_objective_score(x[0],k_iter = 100088)
        #             inital_dep.append(speed_up)
        #             if speed_up > global_best:
        #                 global_best = speed_up
        #                 global_best_indep = inital_indep[idx]
        #             flag = True
        #     ts.append(time.time() - begin)
        #     ss = '{}: step {}, best {}, best-seq {}'.format(str(round(ts[-1])), str(iter + 1),
        #                                                                          str(global_best),
        #                                                                          str(global_best_indep))
        #     write_log(ss, LOG_FILE)
        #     if (time.time() - begin) > time_set_up:
        #         break

        """
        CompTuner
        """
        self.V = self.init_v(len(inital_indep), len(inital_indep[0]), 10, -10)
        for i in range(len(inital_indep)):
            self.pbest.append(0)
            self.p_fit.append(0)
        self.fit = 0
        for i in range(len(inital_indep)):
            self.pbest[i] = inital_indep[i]
            tmp = inital_dep[i]
            self.p_fit[i] = tmp
            if tmp > self.fit:
                self.fit = tmp
                self.gbest = inital_indep[i]
        #
        ts = []  # time spend
        end = time.time()
        ss = '{}: step {}, best {}, cur-best {}, cur-best-seq {}'.format(str(round(end - begin)), str(-1),
                                                                         str(0), str(0), str(0))
        write_log(ss, LOG_FILE)
        begin = time.time()
        for t in range(self.iteration):
            if t == 0:
                self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest,
                                       self.gbest, self.w, self.c1, self.c2, 10, -10)
                for i in range(len(inital_indep)):
                    for j in range(len(inital_indep[0])):
                        a = random.random()
                        if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                            inital_indep[i][j] = 1
                        else:
                            inital_indep[i][j] = 0
                print(inital_indep)

            else:
                merged_predicted_objectives = self.runtime_predict(model, inital_indep)
                for i in range(len(merged_predicted_objectives)):
                    if merged_predicted_objectives[i][1] > self.p_fit[i]:
                        self.p_fit[i] = merged_predicted_objectives[i][1]
                        self.pbest = merged_predicted_objectives[i][0]
                sort_merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
                current_best = sort_merged_predicted_objectives[0][1]
                current_best_seq = sort_merged_predicted_objectives[0][0]
                if current_best > self.fit:
                    self.gbest = current_best_seq
                    self.fit = current_best
                    self.V = self.update_v(self.V, inital_indep, len(inital_indep), len(inital_indep[0]), self.pbest,
                                           self.gbest, self.w, self.c1, self.c2, 10, -10)
                    for i in range(len(inital_indep)):
                        for j in range(len(inital_indep[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-self.V[i][j])) > a:
                                inital_indep[i][j] = 1
                            else:
                                inital_indep[i][j] = 0
                else:
                    avg_dis = 0.0
                    for i in range(1, len(merged_predicted_objectives)):
                        avg_dis = avg_dis + self.getDistance(merged_predicted_objectives[i][0], current_best_seq)
                    print(avg_dis)
                    avg_dis = avg_dis / (len(inital_indep) - 1)
                    print(avg_dis)
                    better_seed_indep = []
                    worse_seed_indep = []
                    better_seed_seq = []
                    worse_seed_seq = []
                    better_seed_pbest = []
                    worse_seed_pbest = []
                    better_seed_V = []
                    worse_seed_V = []
                    #change eight flags
              
                    for i in range(0, len(merged_predicted_objectives)):
                        if self.getDistance(merged_predicted_objectives[i][0], current_best_seq) > avg_dis:
                            worse_seed_indep.append(i)
                            worse_seed_seq.append(merged_predicted_objectives[i][0])
                            worse_seed_pbest.append(self.p_fit[i])
                            worse_seed_V.append(self.V[i])
                        else:
                            better_seed_indep.append(i)
                            better_seed_seq.append(merged_predicted_objectives[i][0])
                            better_seed_pbest.append(self.p_fit[i])
                            better_seed_V.append(self.V[i])
                    """
                    update better particles
                    """
                    V_for_better = self.update_v(better_seed_V, better_seed_seq, len(better_seed_seq),
                                                 len(better_seed_seq[0]), better_seed_pbest, self.gbest
                                                 , self.w, 2 * self.c1, self.c2, 10, -10)
                    for i in range(len(better_seed_seq)):
                        for j in range(len(better_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_better[i][j])) > a:
                                better_seed_seq[i][j] = 1
                            else:
                                better_seed_seq[i][j] = 0
                    """
                    update worse particles
                    """
                    V_for_worse = self.update_v(worse_seed_V, worse_seed_seq, len(worse_seed_seq),
                                                len(worse_seed_seq[0]), worse_seed_pbest, self.gbest
                                                , self.w, self.c1, 2 * self.c2, 10, -10)
                    for i in range(len(worse_seed_seq)):
                        for j in range(len(worse_seed_seq[0])):
                            a = random.random()
                            if 1.0 / (1 + math.exp(-V_for_worse[i][j])) > a:
                                worse_seed_seq[i][j] = 1
                            else:
                                worse_seed_seq[i][j] = 0
                    for i in range(len(better_seed_seq)):
                        inital_indep[better_seed_indep.append[i]] = better_seed_seq[i]
                    for i in range(len(worse_seed_seq)):
                        inital_indep[worse_seed_indep.append[i]] = worse_seed_seq[i]

            print(self.pbest)
            best_result = self.get_objective_score(self.gbest, k_iter=(t + 1))
            ts.append(time.time() - begin)
            ss = '{}: step {}, cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(t + 1),
                                                                             str(self.fit), str(self.gbest))
            write_log(ss, LOG_FILE)
            if (time.time() - begin) > time_set_up:
                break
