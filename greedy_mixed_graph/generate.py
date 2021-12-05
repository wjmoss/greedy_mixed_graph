import numpy as np
import math


def generate_mixed_graph(p, n, iteration=None, p1=0.5, p2=0, p3=0, max_in_degree=np.Inf):
    # n: number of graphs
    # p1: P(directed | empty) = P(empty | directed)
    # 1-p1: P(bidirected | empty) = P(empty | bidirected)
    # p2: P(bidirected | directed) = P(directed | bidirected)
    # p3: P(direction switch | directed)

    # p2 must be < min(p1, 1-p1)
    # p3 must be < 1-p1-p2

    # DAG only version: p1=1

    # Initialize with empty matrix
    mg = np.zeros((p, p)).astype(int)

    # MC iterations
    if iteration is None:
        iteration = 6 * p ** 2
    else:
        iteration = max(6 * p ** 2, iteration)
    res = []
    for k in range((n + 1) * iteration):
        # for some checking?
        # mg_old = mg.copy()

        # pick position uniformly at random
        i = np.random.randint(p)
        j = np.random.randint(p)

        # compute in-degree at the position and its transpose
        in_degree = len(np.where(mg[:, j] > 0)) < max_in_degree
        in_degree_t = len(np.where(mg[:, i] > 0)) < max_in_degree

        # no edge - add directed edge or bidirected edge
        if mg[i, j] == 0 and mg[j, i] == 0 and i != j:

            if np.random.binomial(1, p1) == 1:
                if in_degree:
                    mg[i, j] = 1
            else:
                if in_degree and in_degree_t:
                    mg[i, j] = mg[j, i] = 100

        # directed edge, remove, switch to bidirected, reverse, or stay
        elif mg[i, j] == 1:
            flag = np.random.choice(4, 1, p=[p1, p2, p3, 1 - p1 - p2 - p3]).reshape(())
            if flag == 0:
                mg[i, j] = 0
            elif flag == 1:
                if in_degree and in_degree_t:
                    mg[i, j] = mg[j, i] = 100
            elif flag == 2:
                if in_degree_t:
                    mg[i, j] = 0
                    mg[j, i] = 1

        # bidirected edge - remove, switch to directed, or stay
        elif mg[i, j] == 100:
            flag = np.random.choice(3, 1, p=[1 - p1, p2, p1 - p2]).reshape(())
            if flag == 0:
                mg[i, j] = mg[j, i] = 0
            elif flag == 1:
                if in_degree:
                    mg[i, j] = 1
                    mg[j, i] = 0

        # check if neighbourhood size condition is fulfilled

        if k >= iteration and k % iteration == 0:
            res.append(mg.copy())

    return res


def generate_params(L, O, dist="snormal", posneg=True, Oscale=1):
    # Randomly gendata edge weights and error covariance matrix
    # dist is normal or uniform
    # posneg: boolean, whether or not to use both postive and negetive edge weights
    Lval = L * 1.0
    indL = np.where(L != 0)
    if dist == "snormal":
        Lvals = np.random.normal(0, 1, len(indL[0]))
    else:
        Lvals = np.random.uniform(0.5, 0.9, len(indL[0]))
        if posneg:
            Lvals = (2 * np.random.binomial(1, 0.5, len(indL[0])) - 1) * Lvals
    Lval[indL] = Lvals

    # Repeat generating Omega until minimal eigenvalue is > 1e-6
    while True:
        # self.Oval = (self.O + np.identity(self.p)) * 1.0
        # indO = np.triu_indices(self.p, 1)
        Oval = np.zeros(O.shape)
        indO = np.where(np.triu(O, 1) != 0)
        Oval[indO] = np.random.normal(0, 1, len(indO[0]))
        Oval = Oval + Oval.T

        # set variances to rowsum of abs values plus chi^2(1)
        np.fill_diagonal(Oval, np.sum(Oval, axis=1) + np.random.chisquare(1, O.shape[0]))

        # check minimal eigenvalue
        if min(np.linalg.eigvals(Oval)) > 1e-6: break

    # Rescale source nodes with Oscale
    indices = np.where(np.sum(L, axis=0) == 0)
    Oval[indices, :] = Oval[indices, :] * np.sqrt(Oscale)
    Oval[:, indices] = Oval[:, indices] * np.sqrt(Oscale)
    return Lval, Oval


class SimpleGraph:

    def __init__(self, mg, score=None):
        # mg[i, j] = 1 iff i -> j in the graph
        # mg[i, j] = 100 iff i <-> j in the graph
        # L: directed part
        # O: bidirected part
        # O[i, i] = 1
        mg = mg.copy()
        if not isinstance(mg, np.ndarray) or len(mg.shape) != 2 or mg.shape[0] != mg.shape[1]:
            raise ValueError("Initialization failed, mg must be a square matrix!")
        np.fill_diagonal(mg, 0)
        for i in range(mg.shape[0]):
            for j in range(i + 1, mg.shape[1]):
                if mg[i, j] * mg[j, i] != 0:
                    mg[i, j] = mg[j, i] = 100
                elif mg[i, j] != 0:
                    mg[i, j] = 1
                elif mg[j, i] != 0:
                    mg[j, i] = 1
        self.mg = mg
        self.p = self.mg.shape[0]
        self.L = (self.mg == 1) * 1
        self.O = (self.mg == 100) * 1
        np.fill_diagonal(self.O, 1)
        self._score = score

    # def setmg(self, poslist, values):
    # if len(poslist) != len(values):
    # raise ValueError("Position list and value list must have the same length!")
    # for pos, value in [poslist, values]:
    # if len(pos) < 2:
    # raise ValueError("Position must have length 2!")
    # self.mg[pos[0], pos[1]] = value

    def setmg(self, i, j, val):
        if i == j:
            pass
        if val == 100:
            self.mg[i, j] = self.mg[j, i] = 100
            self.L[i, j] = self.L[j, i] = 0
            self.O[i, j] = self.O[j, i] = 1
        elif val == 0:
            self.mg[i, j] = self.mg[j, i] = 0
            self.L[i, j] = self.L[j, i] = 0
            self.O[i, j] = self.O[j, i] = 0
        else:
            self.mg[i, j] = 1
            self.mg[j, i] = 0
            self.L[i, j] = 1
            self.L[j, i] = 0
            self.O[i, j] = self.O[j, i] = 0

    @property
    def n_edges(self):
        return np.sum(self.mg == 1) + np.sum(self.mg == 100) / 2
        #return len(np.argwhere(self.mg == 1)) + len(np.argwhere(self.mg == 100)) / 2

    def split(self):
        L = self.mg.copy()
        O = self.mg.copy()
        L[L == 100] = 0
        O[O == 1] = 0
        O[O == 100] = 1
        np.fill_diagonal(O, 1)
        return L, O

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    def get_skeleton(self):
        sk = ((self.mg + self.mg.T) > 0) * 1
        return sk

    def get_vstructures(self):
        pass

    def getColliderInvariantModels(self):
        pass


#class ScoredSimpleGraph(SimpleGraph):
#    def __init__(self, mg, score):
#        super(ScoredSimpleGraph, self).__init__(mg)
#        self.score = score

#    def setScore(self, newscore):
#        self.score = newscore


class ParamedSimpleGraph(SimpleGraph):
    def __init__(self, mg):
        super(ParamedSimpleGraph, self).__init__(mg)
        self.Lval = None
        self.Oval = None
        self.data = None

    def assignParams(self, Lval, Oval):
        if not isinstance(Lval, np.ndarray) or not isinstance(Oval, np.ndarray):
            raise ValueError("L and O must be matrices!")
        if Lval.shape != (self.p, self.p) or Oval.shape != (self.p, self.p):
            raise ValueError("L and O must have the same size as mg!")
        self.Lval = Lval * 1.0 * self.L

        # make Oval symmetric and check positive definite
        Oval = (Oval + Oval.T) / 2.0
        self.Oval = Oval * self.O
        if min(np.linalg.eigvals(self.Oval)) <= 1e-6:
            raise ValueError("O must be positive definite!")

    @property
    def sigma(self):
        # compute true covariance matrix, given model parameters L and Omega
        IminusL = np.identity(self.p) - self.Lval
        return np.linalg.inv(IminusL).T @ self.Oval @ np.linalg.inv(IminusL)

    @property
    def llh(self):
        #log-likelihood / n
        n = self.data.shape[0]
        return -0.5 * (self.p * math.log(2 * math.pi) + math.log(np.linalg.det(self.sigma)) +
                       (n - 1) / n * sum(np.diag(np.linalg.inv(self.sigma) @ np.cov(self.data.T))))

    def isFaithful(self, faithful_eps):
        indL = np.where(self.L != 0)
        indO = np.where(np.triu(self.O, 1) != 0)
        if np.any(abs(self.Lval[indL]) < faithful_eps) or np.any(abs(self.Oval[indO]) < faithful_eps):
            return False
        else:
            return True

    def generateData(self, n):
        # np multivare normal result is n * p
        # data is n * p
        eps = np.random.multivariate_normal(mean=np.repeat(0, self.p), cov=self.Oval, size=n)
        self.data = eps @ np.linalg.inv(np.identity(self.p) - self.Lval)


class EstimatedGraph(ParamedSimpleGraph):
    def __init__(self, mg):
        #tbd
        super(EstimatedGraph, self).__init__(mg)


if __name__ == '__main__':
    #np.random.seed(19260817)
    graphs = generate_mixed_graph(p=4, n=10)
    print(graphs)
    g = ParamedSimpleGraph(graphs[0])
    params = generate_params(L=g.L, O=g.O)
    g.assignParams(params[0], params[1])
    g.generateData(1000000)
    # print(g.getSigma())
    # print(g.data.shape)
    # print(np.cov(g.data.T))
    print(np.max(abs(np.cov(g.data.T) - g.sigma)))