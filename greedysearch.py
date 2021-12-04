import numpy as np
import math
import time
import copy
import multiprocessing as mp
from functools import partial

from ricf import ricf
from generate import SimpleGraph, generateParams, generateMixedGraph, ParamedSimpleGraph

set_edge_penalty = 1


class SearchResult:

    def __init__(self, final_graph, final_score, all_graphs, all_scores, times):
        self.final_graph = final_graph
        self.final_score = final_score
        self.all_graphs = all_graphs
        self.all_scores = all_scores
        # cumtimes
        self.times = times

    def printinfo(self):
        print("final graph:")
        print(self.final_graph)
        print("final score:")
        print(self.final_score)
        print("All graphs:")
        print(self.all_graphs)
        print("All scores:")
        print(self.all_scores)
        print("times:")
        print(self.times)


class GreedySearch:
    # mg_start: simple mixed graph, with entries 1/100
    # can be a np.ndarray type matrix, a list of matrices, an SimpleGraph object, a list of SimplGraph objects
    # n: sample size
    # cov_mat: sample covariance matrix
    # max_steps: maximal number of greedy search steps
    # direction: control forward/backward search, edge type change
    # max_iter: maximal number of iterations in ricf
    # edge_penalty: hyperparameter lambda
    # verbose: print info
    # faithful_eps: faithful eps in ricf
    # max_pos: maximal number of positions for adding/deleting/edge changing
    # dags_only: only directed graph?
    # eps_conv: eps for greedy search
    # reinitialization: restarts of ricf, default 0
    # bic: "std_bic" or "ext_bic" (default)

    def __init__(self, n, cov_mat, mg_start=None, max_steps=np.Inf, direction=2, max_iter=20,
                 edge_penalty=set_edge_penalty, verbose=False, faithful_eps=0, max_pos=np.Inf,
                 dags_only=False, eps_conv=1e-12, reinitialization=0, bic="std_bic"):
        self.n = n
        self.cov_mat = cov_mat

        if isinstance(mg_start, SimpleGraph):
            self.mg_start = mg_start
        elif isinstance(mg_start, np.ndarray) and mg_start.shape[0] == mg_start.shape[1]:
            self.mg_start = SimpleGraph(mg_start)
        elif isinstance(mg_start, list):
            self.mg_start = list(map(SimpleGraph, mg_start))
        else:
            self.mg_start = None
            #self.mg_start = SimpleGraph(np.zeros((p, p)).astype(int))
            #raise TypeError("mg_start must be a SimpleGraph object or a square matrix!")

        self.max_steps = max_steps
        self.direction = direction
        self.max_iter = max_iter
        self.edge_penalty = edge_penalty
        self.verbose = verbose
        self.faithful_eps = faithful_eps
        self.max_pos = max_pos
        self.dags_only = dags_only
        self.eps_conv = eps_conv
        self.reinitialization = reinitialization
        self.bic = bic

    def compute_score(self, mg: SimpleGraph):
        p = mg.p
        L, O = mg.split()

        if self.reinitialization == 0:
            flag = 0
            try:
                res = ricf(S=self.cov_mat, O=O, L=L, sigconv=False, tol=1e-6,
                                maxiter=self.max_iter, out="none", maxkap=1e9)
            except Exception as e:
                flag = 1
                print('Exception:', e)
            finally:
                if flag:
                    return np.NINF

            # bic
            if self.bic == "std_bic":
                score = -0.5 * (p * math.log(2 * math.pi) + math.log(np.linalg.det(res.sigest)) +
                                (self.n - 1) / self.n * sum(np.diag(np.linalg.inv(res.sigest) @ self.cov_mat))) - \
                        self.edge_penalty * math.log(self.n) / 2 / self.n * (mg.n_edges + p)
            # ext_bic
            else:
                score = -0.5 * (p * math.log(2 * math.pi) + math.log(np.linalg.det(res.sigest)) +
                                (self.n - 1) / self.n * sum(np.diag(np.linalg.inv(res.sigest) @ self.cov_mat))) - \
                        self.edge_penalty * math.log(self.n) / 2 / self.n * (mg.n_edges + p) - \
                        (mg.n_edges + p) * (2 * math.log(p) + math.log(3)) / self.n
            maxscore = score

        else:
            scores = []
            for i in range(self.reinitialization):
                Linit, Oinit = generateParams(L, O)
                flag = 0
                try:
                    res = ricf(S=self.cov_mat, O=O, L=L, Linit=Linit, Oinit=Oinit, sigconv=False, tol=1e-6,
                                    maxiter=self.max_iter, out="none", maxkap=1e9)
                except Exception as e:
                    flag = 1
                    print('Exception:', e)
                finally:
                    if flag:
                        scores.append(np.NINF)
                    else:
                        # bic
                        if self.bic == "std_bic":
                            score = -0.5 * (p * math.log(2 * math.pi) + math.log(np.linalg.det(res.sigest)) +
                                            (self.n - 1) / self.n * sum(np.diag(np.linalg.inv(res.sigest) @ self.cov_mat))) - \
                                    self.edge_penalty * math.log(self.n) / 2 / self.n * (mg.n_edges + p)
                        # ext_bic
                        else:
                            score = -0.5 * (p * math.log(2 * math.pi) + math.log(np.linalg.det(res.sigest)) +
                                            (self.n - 1) / self.n * sum(
                                        np.diag(np.linalg.inv(res.sigest) @ self.cov_mat))) - \
                                    self.edge_penalty * math.log(self.n) / 2 / self.n * (mg.n_edges + p) - \
                                    (mg.n_edges + p) * (2 * math.log(p) + math.log(3)) / self.n
                        scores.append(score)
            maxscore = max(scores)
        return maxscore

    def greedy_search_one_rep(self, mg:SimpleGraph):
        k = 1
        t = time.time()
        p = self.cov_mat.shape[0]

        # initial score
        state = mg
        score = self.compute_score(mg)
        state.score = score
        states = []
        times = [time.time() - t]
        cumtimes = [times[0]]

        while k < self.max_iter:
            # 0 -- only forward
            # 1 -- only backward
            # 2 -- both
            # <=2 do edge type change

            # forward
            cand_add = []
            if self.direction != 1:
                poslist = np.where(np.triu(state.mg + 1, 1) + np.triu(state.mg.T + 1, 1) == 2)
                indlist = np.random.choice(len(poslist[0]), min(self.max_pos, len(poslist[0])), replace=False)
                for ind in indlist:
                    i = poslist[0][ind]
                    j = poslist[1][ind]

                    if not self.dags_only:
                        # add bidirected edge
                        newstate = copy.deepcopy(state)
                        newstate.setmg(i, j, 100)
                        newstate.setmg(j, i, 100)
                        newstate.score = self.compute_score(newstate)
                        cand_add.append(newstate)

                    # add directed edge, two directions
                    newstate = copy.deepcopy(state)
                    newstate.setmg(i, j, 1)
                    newstate.score = self.compute_score(newstate)
                    cand_add.append(newstate)

                    newstate = copy.deepcopy(state)
                    newstate.setmg(j, i, 1)
                    newstate.score = self.compute_score(newstate)
                    cand_add.append(newstate)

            # backward
            cand_del = []
            if self.direction != 0:
                poslist = np.where(np.triu(state.mg, 1) + np.triu(state.mg.T, 1) != 0)
                indlist = np.random.choice(len(poslist[0]), min(self.max_pos, len(poslist[0])), replace=False)
                for ind in indlist:
                    i = poslist[0][ind]
                    j = poslist[1][ind]

                    # delete edge
                    newstate = copy.deepcopy(state)
                    newstate.setmg(i, j, 0)
                    newstate.setmg(j, i, 0)
                    newstate.score = self.compute_score(newstate)
                    cand_del.append(newstate)

            # edge type change
            cand_cha = []
            if self.direction <= 2:
                # traverse directed edges
                poslist = np.where(state.mg == 1)
                indlist = np.random.choice(len(poslist[0]), min(self.max_pos, len(poslist[0])), replace=False)
                for ind in indlist:
                    i = poslist[0][ind]
                    j = poslist[1][ind]

                    # change to bidirected edge
                    if not self.dags_only:
                        newstate = copy.deepcopy(state)
                        newstate.setmg(i, j, 100)
                        newstate.setmg(j, i, 100)
                        newstate.score = self.compute_score(newstate)
                        cand_cha.append(newstate)

                    # reverse direction
                    newstate = copy.deepcopy(state)
                    newstate.setmg(i, j, 0)
                    newstate.setmg(j, i, 1)
                    newstate.score = self.compute_score(newstate)
                    cand_cha.append(newstate)

                # traverse bidirected edges
                poslist = np.where(np.triu(state.mg) == 100)
                indlist = np.random.choice(len(poslist[0]), min(self.max_pos, len(poslist[0])), replace=False)
                for ind in indlist:
                    i = poslist[0][ind]
                    j = poslist[1][ind]

                    # 1 in upper tri
                    newstate = copy.deepcopy(state)
                    newstate.setmg(i, j, 1)
                    newstate.setmg(j, i, 0)
                    newstate.score = self.compute_score(newstate)
                    cand_cha.append(newstate)

                    # 1 in lower tri
                    newstate = copy.deepcopy(state)
                    newstate.setmg(i, j, 0)
                    newstate.setmg(j, i, 1)
                    newstate.score = self.compute_score(newstate)
                    cand_cha.append(newstate)

            # compare candidates and select best one
            if len(cand_add) > 0:
                best_a = np.argmax(list(map(lambda x: x.score, cand_add))).reshape(())
                best_a_score = cand_add[best_a].score
            else:
                best_a = None
                best_a_score = np.NINF

            if len(cand_del) > 0:
                best_d = np.argmax(list(map(lambda x: x.score, cand_del))).reshape(())
                best_d_score = cand_del[best_d].score
            else:
                best_d = None
                best_d_score = np.NINF

            if len(cand_cha) > 0:
                best_c = np.argmax(list(map(lambda x: x.score, cand_cha))).reshape(())
                best_c_score = cand_cha[best_c].score
            else:
                best_c = None
                best_c_score = np.NINF

            states.append(state)

            # break out the loop if no improvement of score
            if max([best_a_score, best_d_score, best_c_score]) <= state.score + self.eps_conv:
                break

            # otherwise change state to the best candidate
            ind = np.argmax([best_a_score, best_d_score, best_c_score]).reshape(())
            if ind == 0:
                state = cand_add[best_a]
            elif ind == 1:
                state = cand_del[best_d]
            else:
                state = cand_cha[best_c]

            # take time for step
            times.append(time.time() - t)
            cumtimes.append(cumtimes[-1] + times[-1])
            t = time.time()

            # info output
            action = ["ADD", "DEL", "CHA"][ind]
            lengths = "(" + str(len(cand_add)) + ", " + str(len(cand_del)) + ", " + str(len(cand_cha)) + ")"
            if self.verbose:
                print(k, action, "; (adds, dels, changes) = ", lengths, "; steptime =", times[-1], "; score =",
                      state.score)

            k += 1
            if k >= self.max_steps:
                print("MAX STEPS ACHIEVED")

        return state, states, times, cumtimes

    def greedy_search(self, n_restarts, mc_cores):
        if self.mg_start is None:
            p1 = 1 if self.dags_only else 0.5
            mg_stash = generateMixedGraph(p=self.cov_mat.shape[0], N=n_restarts, p1=p1, max_in_degree=1)
            mg_stash = list(map(SimpleGraph, mg_stash))
        elif isinstance(self.mg_start, SimpleGraph):
            mg_stash = [self.mg_start] * n_restarts
        elif isinstance(self.mg_start, list):
            ind = np.random.choice(len(self.mg_start), n_restarts, replace=False)
            mg_stash = self.mg_start[np.random.choice(len(self.mg_start), n_restarts, replace=False)]
        else:
            raise TypeError("Invalid mg_start type!")

        #direction_map = {"forward": 0, "backward": 1, "both": 2}

        if mc_cores > 1:
            pool = mp.Pool(min(mp.cpu_count(), mc_cores))
            res = [pool.apply_async(self.greedy_search_one_rep, args=(mg_stash[i],)) for i in range(n_restarts)]
            res = [obj.get() for obj in res]
            pool.close()
        else:
            res = list(map(self.greedy_search_one_rep, [mg_stash[i] for i in range(n_restarts)]))

        i_best = np.argmax(list(map(lambda x: x[0].score, res))).reshape(())
        final_graph = res[i_best][0]
        final_score = res[i_best][0].score
        all_graphs = list(map(lambda x: list(map(lambda state: state.mg, x[1])), res))
        all_scores = list(map(lambda x: list(map(lambda state: state.score, x[1])), res))
        times = list(map(lambda x: x[3], res))

        return SearchResult(final_graph, final_score, all_graphs, all_scores, times)


if __name__ == '__main__':
    np.random.seed(19260817)
    graphs = generateMixedGraph(p=5, N=10)
    g = ParamedSimpleGraph(graphs[0])
    params = generateParams(L=g.L, O=g.O)
    g.assignParams(params[0], params[1])
    g.generateData(10000)
    Glist = [g.mg] * 10
    GS = GreedySearch(mg_start=g.mg, n=10000, cov_mat=np.cov(g.data.T), bic='std_bic')
    GS = GreedySearch(n=10000, cov_mat=np.cov(g.data.T), bic='ext_bic')
    result = GS.greedy_search(n_restarts=10, mc_cores=2)
    print('hhh')
    print(g.mg)
    print(result.final_graph.mg)
