# INPUTS ---
# L, O: the 0-1 valued adjacency matrices indicating the structure of the graph G
# S: the p-by-p covariance matrix of observations from the true model
# Linit; Oinit: the initial values of the edge parameters for running the algorithm
# sigconv: Boolean (True = look for conv. in Sigma, False = look for conv. in L/O )
# tol: gives the max average entrywise absolute deviation in Sigma permited for convergence
# maxiter: gives the max number of iterations before divergence is accepted
# out: String: options: None/False, Final, All/True
# maxkap: maximum condition number accepted before error thrown
# B: (optional instead of L -- here B = t(L))
#
# OUTPUTS ---
# Sigmahat: the fitted value for Sigma resulting from the algorithm
# Bhat: the fitted value for the directed edge matrix (note: equal to I-t(Lambdahat))
# Omegahat, Lambdahat: the MLE for the parameter values resulting from the algorithm
# iterations: the number of iterations run by the algorithm before convergence or divergence accepted
# converged: TRUE or FALSE - based on whether or not the algorithm converged before maxiter reached
import warnings
import numpy as np

class Estimate():
    def __init__(self, sigcur, lhat, Ocur, Lcur, iteration, converged):
        self.sigest = sigcur
        self.lhat = lhat
        self.Oest = Ocur
        self.Lest = Lcur
        self.iteration = iteration
        self.converged = converged


#ricf_covMat
def ricf(S, O, L = None, Linit=None, Oinit=None, sigconv=True, tol=1e-6,
            maxiter=5000, out='none', maxkap=1e13, B=None):
    if L is None:
        if not isinstance(B,np.ndarray): raise ValueError("B must be a matrix!")
        if B.ndim != 2 or B.shape[0] != B.shape[1]: raise ValueError("B must be a square matrix!")
        L = B.T
    if not isinstance(L, np.ndarray) or not isinstance(O, np.ndarray) \
            or L.ndim != 2 or O.ndim != 2:
        raise ValueError("L and O must be matrices!")
    if L.shape[0] != L.shape[1] or L.shape[0] != O.shape[0] or O.shape[0] != O.shape[1]:
        raise ValueError("L and O must be square matrices of the same size!")
    p = L.shape[0]
    if not isinstance(S, np.ndarray): raise ValueError("S must be a matrix!")
    if S.shape[0] != S.shape[1] or S.shape[0] != L.shape[0]:
        raise ValueError("S must be a square matrix the same size as L!")

    # initialize the directed edge weights via OLS
    def initL(L, S):
        Linit = L.copy() * 1.0
        # change to list comprehension
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                if L[i, j] != 0:
                    Linit[i, j] = S[i, j] / S[i, i]
        #Linit = [[S[i, j] * 1.0 / S[i, i] for j in range(L.shape[1])] for i in range(L.shape[0])]
        return Linit

    # Initialize the bidirected edge parameters at random
    def initO(O, S):
        s = min(np.diagonal(S))
        R = np.diag(np.diagonal(S)) * 1.0
        # change to list comprehension
        if p > 1:
            for i in range(p):
                for j in range(i+1,p):
                    R[i, j] = R[j, i] = np.random.uniform(0, 0.5, 1) * s
        #R = np.random.uniform(0, 0.5, O.shape[0]*O.shape[0]).reshape(O.shape[0], -1)
        #R = np.triu(R, 1)
        O2 = R * (O != 0)
        np.fill_diagonal(O2, np.diagonal(S))
        lm = min(np.linalg.eigvals(O2))
        if lm < 0:
            np.fill_diagonal(O2, np.diagonal(S) + abs(lm) + 0.1)
        return O2

    if Linit is None: Linit = initL(L, S)
    if Oinit is None: Oinit = initO(O, S)
    if not isinstance(Linit, np.ndarray) or not isinstance(Oinit, np.ndarray):
        raise ValueError("Linit and Oinit must be matrices!")
    if Linit.ndim != 2 or Linit.shape[0] != Linit.shape[1] or Oinit.ndim != 2 or Oinit.shape[0] != Oinit.shape[1]:
        raise ValueError("Linit and Oinit must be square matrices!")
    if len(np.unique([Linit.shape[0], Oinit.shape[0], L.shape[0]])) > 1:
        raise ValueError("one of the input matrices has the wrong dimension!")
    if np.sum(abs(L * O)) != 0:
        warnings.warn("The graph either contains a bow or a self-loop.", UserWarning)
    if maxiter <= 0 or maxiter % 1 != 0:
        raise ValueError("A positive integer is needed for the max number of iterations!")
    if tol <= 0:
        raise ValueError("A positive tolerance is needed for convergence to be possible!")
    if type(sigconv) != bool:
        raise ValueError("sigconv needs to take on a logical value!")
    if not(out == "true" or out == "all" or out == "final" or out == "false" or out == "none"):
        raise ValueError("Output variable needs to be: none/false, final, or all/true!")

    def Det(Lcur):
        return np.linalg.det(np.identity(Lcur.shape[0])-Lcur)

    def tarjan(u, dft, low, count, visited, stack):
        dft[u] = low[u] = count
        count += 1
        visited[u] = 1
        stack.append(u)
        # np.where returns a tuple?
        edgelist = np.where(L[u, :] != 0)[0]
        #edgelist = np.argwhere(L[u, :] != 0)
        for v in edgelist:
            if not visited[v]:
                tarjan(v, dft, low, count, visited, stack)
                low[u] = min(low[u], low[v])
            elif v in stack:
                low[u] = min(low[u], dft[v])
        if dft[u] == low[u]:
            while stack.pop() != u:
                visited[u] += 1

    Lcur, Ocur = Linit, Oinit
    iteration = 1
    if p > 1:
        # determine the nodes that only need 1 iteration
        # has no spouses and not in a cycle
        # not in a cycle: to find strongly connected components via Tarjan's algorithm
        visited = [0] * p
        dft = [0] * p
        low = [0] * p
        count = 0
        stack = []
        for u in range(p):
            if not visited[u]:
                tarjan(u, dft, low, count, visited, stack)

        while True:
            for i in range(p):
                # O[i, -i] is zero vector means no spouses
                # dft[i] == low[i] and visited[i] == 1 means the node is a component itself
                if iteration > 1:
                    if sum(O[i, :]) == 1 and dft[i] == low[i] and visited[i] == 1:
                        continue
                #pa = np.argwhere(L[:, i] != 0)
                pa = np.where(L[:, i] != 0)[0]
                n_pa = len(pa)
                sp = np.where(O[:, i] != 0)[0]
                sp = sp[sp != i]
                n_sp = len(sp)
                totalLen = n_pa + n_sp
                IminusL = np.identity(p) - Lcur
                ind_minusI = [x for x in range(p) if x != i]
                Elessi = np.delete(IminusL, i, axis=1)

                if np.linalg.cond(Ocur[np.ix_(ind_minusI, ind_minusI)], p=2) > maxkap:
                    raise ValueError("The condition number of Ocur[-i, -i] is too large for node " + str(i))
                Zlessi = np.matmul(Elessi, np.linalg.inv(Ocur[np.ix_(ind_minusI, ind_minusI)]))
                Zhelp = np.matmul(S, Zlessi)
                # The following line gets the indices of Zlessi corresponding to spouses
                zsp = np.concatenate((sp[sp < i], sp[sp > i] - 1), axis=None)

                if totalLen > 0:
                    if n_pa > 0:
                        # DETERMINE a AND a0
                        # a: coefficient of B[i,pa(i)]/L[pa(i),i]
                        # a0: const independent to B[i,pa(i)]/L[pa(i),i]
                        a = np.repeat(0.0, n_pa, axis=None)
                        for k in range(n_pa):
                            temp = Lcur.copy()
                            temp[pa[k], i] = 2
                            det2 = Det(temp)
                            temp[pa[k], i] = 1
                            det1 = Det(temp)
                            a[k] = det2 - det1
                        temp[pa, i] = 0
                        a0 = Det(temp)

                        ind_pos = np.where(a != 0)
                        pa_pos = pa[ind_pos]

                        if n_sp == 0:
                            M = S[np.ix_(pa, pa)]
                            m = S[np.ix_(pa, [i])]
                        # n_sp > 0
                        else:
                            ZtZ = np.matmul(Zlessi.T, Zhelp)[np.ix_(zsp, zsp)]
                            ZtY = Zhelp.T[np.ix_(zsp, pa)]
                            YtY = S[np.ix_(pa, pa)]
                            M = np.concatenate(
                                (np.concatenate((ZtZ, ZtY), axis=1), np.concatenate((ZtY.T, YtY), axis=1)), axis=0)
                            m = np.concatenate((Zhelp.T[np.ix_(zsp, [i])], S[np.ix_(pa, [i])]), axis=0)

                        alpha = np.linalg.solve(M, m)
                        if np.any(np.isnan(alpha)):
                            raise ValueError("Collinearity observed for node " + str(i))
                        y0 = S[i, i] - np.matmul(alpha.T, m)
                        if len(ind_pos) > 0:
                            # general case
                            coef = alpha + (y0 / (a0 + np.matmul(a.T, alpha[n_sp:totalLen]))) * np.linalg.solve(M, np.concatenate((np.repeat(0.0, n_sp), a)).reshape(-1,1))
                        else:
                            # a = 0, reduced to Linear Regression
                            coef = alpha
                        coef = coef.reshape(-1)
                        if n_sp > 0: Ocur[i, sp] = Ocur[sp, i] = coef[:n_sp]
                        if len(coef) > n_sp + n_pa:
                            print(i)
                            print(len(coef), n_pa + n_sp)
                            raise Exception("Angry!")
                        Lcur[pa, i] = coef[n_sp:] if n_pa > 0 else 0.0
                    else:
                        # no parents, only spouses
                        # Solve a simplified version with no Betas (Only Omegas)
                        M = ZtZ = np.matmul(Zlessi.T, Zhelp)[np.ix_(zsp, zsp)]
                        m = Zhelp.T[np.ix_(zsp, [i])]
                        coef = np.linalg.solve(M, m).reshape(-1)
                        Ocur[sp, i] = Ocur[i, sp] = coef
                        y0 = S[i, i] - np.matmul(coef.T, m)

                    # Find the variance omega_{ii}
                    RSS_avg = S[i, i] - 2 * np.matmul(coef.T, m) + coef.T @ M @ coef
                    if np.linalg.cond(Ocur[np.ix_(ind_minusI, ind_minusI)], p=2) > maxkap:
                        raise ValueError("The condition number of Ocur[-i, -i] is too large for node " + str(i))

                    ## print ##
                    Ocur[i, i] = RSS_avg + Ocur[np.ix_([i], ind_minusI)] @ np.linalg.inv(Ocur[np.ix_(ind_minusI, ind_minusI)]) @ Ocur[np.ix_(ind_minusI, [i])]
                else:
                    Ocur[i, i] = S[i, i]
            lhat = np.identity(p) - Lcur
            sigcur = np.linalg.inv(lhat).T @ Ocur @ np.linalg.inv(lhat)
            if iteration == 1:
                if out == "true" or out == "all": print(str(iteration) + '\n')
                if maxiter == 1:
                    if out in ["true", "all", "final"]:
                        print("Sigmahat")
                        print(sigcur)
                        print("nLhat")
                        print(lhat)
                        print("nOmegahat")
                        print(Ocur)
                        print("nLambdahat")
                        print(Lcur)
                        print("niterations")
                        print(iteration)
                    break
            elif iteration > 1:
                dsig = np.mean(abs(sigcur - sigpast))
                dLO = np.sum(abs(Lcur - Lpast) + abs(Ocur - Opast)) / (np.sum(L) + np.sum(O))
                if out == "true" or out == "all":
                    dsig6 = round(dsig, 6)
                    dLO6 = round(dLO, 6)
                    print("{} Avg Change in L & O: {} | Avg Vhange in Sigma: {}".format(iteration, dLO6, dsig6))
                if (sigconv and dsig < tol) or (not sigconv and dLO < tol) or (iteration >= maxiter):
                    if out in ["true", "all", "final"]:
                        print("Sigmahat")
                        print(sigcur)
                        print("nLhat")
                        print(lhat)
                        print("nOmegahat")
                        print(Ocur)
                        print("nLambdahat")
                        print(Lcur)
                        print("niterations")
                        print(iteration)
                    break

            sigpast = sigcur.copy()
            Lpast = Lcur.copy()
            Opast = Ocur.copy()
            iteration = iteration + 1

    else:
        sigcur = np.array(S[p, p])
        lhat = np.array(0)
        Ocur = np.array(S[p, p])
        Lcur = np.array(0)
        iteration = 1

    converged = iteration < maxiter
    #return sigcur, lhat, Ocur, Lcur, iteration, converged
    return Estimate(sigcur, lhat, Ocur, Lcur, iteration, converged)
