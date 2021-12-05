# 执行测试用例方案一如下：
# unittest.main()方法会搜索该模块下所有以test开头的测试用例方法，并自动执行它们。

import unittest
import numpy as np
from greedy_mixed_graph.generate import *
from greedy_mixed_graph.ricf import *
from greedy_mixed_graph.greedysearch import *


# 定义测试类，父类为unittest.TestCase。
# 可继承unittest.TestCase的方法，如setUp和tearDown方法，不过此方法可以在子类重写，覆盖父类方法。
# 可继承unittest.TestCase的各种断言方法。
class Test(unittest.TestCase):

    def setUp(self):
        pass
        #self.number = raw_input('Enter a number:')
        #self.number = int(self.number)

    def tearDown(self) -> None:
        pass

    def test_generate(self):
        np.random.seed(19260817)
        graphs = generate_mixed_graph(p=5, n=10)
        # check graphs
        self.assertIsInstance(graphs, list)
        map(lambda x: self.assertIsInstance(x, np.ndarray), graphs)
        map(lambda x: self.assertEqual(x.shape[0], x.shape[1]), graphs)

        g = ParamedSimpleGraph(graphs[0])
        # check simple graph
        self.assertTrue((1 - g.L * g.O).all())\
        #self.assertFalse(np.min(g.L * g.O))
        self.assertTrue((1 - g.L * (g.L - 1)).all())
        self.assertTrue((1 - g.O * (g.O - 1)).all())

        params = generate_params(L=g.L, O=g.O)
        g.assignParams(params[0], params[1])

        # check sample covariance matrix
        g.generateData(1000000)
        tol = 0.05
        self.assertTrue(np.max(abs(g.sigma - np.cov(g.data.T))) < tol)

    def test_ricf(self):
        g = ParamedSimpleGraph(generate_mixed_graph(p=5, n=1)[0])
        params = generate_params(L=g.L, O=g.O)
        g.assignParams(params[0], params[1])
        g.generateData(1000000)
        tol = 0.05
        try:
            res = ricf(S=np.cov(g.data.T), O=g.O, L=g.L, sigconv=False, tol=1e-6,
                       maxiter=50, out="none", maxkap=1e9)
        except Exception as e:
            print('Exception:', e)
        if res.converged:
            self.assertTrue(np.max(abs(res.sigest - g.sigma)) < tol)

    def test_search(self):
        graphs = generate_mixed_graph(p=5, n=10)
        g = ParamedSimpleGraph(graphs[0])
        params = generate_params(L=g.L, O=g.O)
        g.assignParams(params[0], params[1])
        g.generateData(100000)
        mg_start = list([g.mg, graphs[-1]])
        GS = GreedySearch(mg_start=mg_start, n=100000, cov_mat=np.cov(g.data.T), bic='std_bic')
        res = GS.greedy_search(n_restarts=2, mc_cores=2)
        tol = 0.01
        # check the final score is not smaller than the score of ground truth (log-likelihood - penalty)
        self.assertTrue(res.final_score >= res.all_scores[0][0])


if __name__ == '__main__':
    unittest.main()