import scipy
import cvxopt
import logging
import numpy as np
from enum import Enum

def order_transformation(data, param_num, precision=np.float32):
    rows, cols = data.shape[0], param_num
    phi = np.zeros([rows, cols], dtype=precision)
    
    for idx, x in enumerate(data):
        if isinstance(x, np.ndarray):
            assert len(x) == param_num, 'order transformation: sample dim must to equal to params number'
        else:
            x = np.repeat(x, param_num)

        phi_x = []
        for k in range(0, cols):
            phi_x.append(x[k] ** k)
        phi[idx, ...] = phi_x
    return phi


def identity_transformation(data, param_num, precision=np.float32):
    assert param_num == data.shape[1]
    rows, cols = data.shape[0], data.shape[1]
    phi = np.zeros([rows, cols], dtype=precision)

    for idx, x in enumerate(data):
        phi_x = []
        for k in range(0, cols):
            phi_x.append(x[k])
        phi[idx, ...] = phi_x
    return phi


def order_2nd_transformation(data, param_num, precision=np.float32):
    assert param_num == 2 * data.shape[1]
    rows, cols = data.shape[0], param_num
    phi = np.zeros([rows, cols], dtype=precision)

    for idx, x in enumerate(data):
        phi_x = []
        for order in range(1, 3):
            for k in range(0, data.shape[1]):
                phi_x.append(x[k] ** order) 

        phi[idx, ...] = phi_x
    return phi           


def order_4nd_transformation(data, param_num, precision=np.float32):
    assert param_num == 4 * data.shape[1]
    rows, cols = data.shape[0], param_num
    phi = np.zeros([rows, cols], dtype=precision)

    for idx, x in enumerate(data):
        phi_x = []
        for order in range(1, 5):
            for k in range(0, data.shape[1]):
                phi_x.append(x[k] ** order) 

        phi[idx, ...] = phi_x
    return phi 


ALGO_TYPE = Enum('ALGO_TYPE', ('BASE', 'LS', 'RLS', 'LASSO', 'RR', 'BR'))


class Algorithm:
    def __init__(self, transformation, param_num, name='', algo_type=ALGO_TYPE.BASE):
        self.transformation = transformation
        self.param_num = param_num
        self._param= np.ones(param_num)
        self.name=name
        self.algo_type = algo_type

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, val):
        self._param = val

    def regress(self, x, y):
        y = np.transpose(y)
        phi = self.transformation(x, self.param_num)
        phi = np.transpose(phi)
        return y, phi

    def predict(self, x):
        phi = self.transformation(x, self.param_num)
        return np.dot(phi, np.transpose(self._param))
    
    def mse_loss(self, x, y):
        prediction = self.predict(x)
        return np.mean((y - prediction) ** 2)

    def objective_loss(self, x, y):
        return NotImplementedError


class LeastSquares(Algorithm):
    def __init__(self, transformation, param_num, name=''):
        super().__init__(transformation, param_num, name, algo_type=ALGO_TYPE.LS)
    
    def regress(self, x, y):
        y, phi = super().regress(x, y)

        tmp = np.dot(phi, np.transpose(phi))    
        tmp = np.linalg.inv(tmp)
        tmp = np.dot(tmp, phi)
        self._param= np.dot(tmp, y)
        return self._param
    
    def objective_loss(self, x, y):
        return self.mse_loss(x, y)


class RegularizedLS(Algorithm):
    def __init__(self, transformation, param_num, name='', alpha=0):
        name = name + '_' + str(alpha)
        super().__init__(transformation, param_num, name, algo_type=ALGO_TYPE.RLS)
        self.alpha = alpha

    def regress(self, x, y):
        y, phi = super().regress(x, y)

        tmp = np.dot(phi, np.transpose(phi))
        tmp += self.alpha * np.eye(tmp.shape[1])
        tmp = np.linalg.inv(tmp)
        tmp = np.dot(tmp, phi)
        self._param= np.dot(tmp, y)
        return self._param

    def objective_loss(self, x, y):
        mse = self.mse_loss(x, y)
        return mse + self.alpha * np.sum(self._param ** 2)


class L1RegularizedLS(Algorithm):
    def __init__(self, transformation, param_num, name='', alpha=0):
        name = name + '_' + str(alpha)
        super().__init__(transformation, param_num, name, algo_type=ALGO_TYPE.LASSO)
        self.alpha = alpha

    def regress(self, x, y):
        y, phi = super().regress(x, y)

        # create H
        p = np.dot(phi, np.transpose(phi))
        P1 = np.hstack((p, -p))
        P2 = np.hstack((-p, p))
        P = np.vstack((P1, P2))

        # create f
        tmp = np.dot(phi, y)
        tmp = np.hstack((tmp, -tmp))
        q = self.alpha * np.ones(tmp.shape) - tmp
        # create G and T
        G = -np.eye(q.shape[0])
        h = np.zeros(q.shape[0])

        # qp solver
        P = cvxopt.matrix(P.tolist())
        q = cvxopt.matrix(q.tolist())
        G = cvxopt.matrix(G.tolist())
        h = cvxopt.matrix(h.tolist())
        result = cvxopt.solvers.qp(P, q, G, h)
        theta = np.array(list(result['x']))
        positive = theta[:self.param_num]
        negtive = theta[self.param_num:]

        self._param= positive - negtive
        return self._param

    def objective_loss(self, x, y):
        mse = self.mse_loss(x, y)
        return mse + self.alpha * np.sum(np.abs(self._param))


class RobustRegression(Algorithm):
    def __init__(self, transformation, param_num, name=''):
        super().__init__(transformation, param_num, name, algo_type=ALGO_TYPE.RR)
    
    def regress(self, x, y):
        y, phi = super().regress(x, y)
        # create c
        c1 = np.zeros(self.param_num)
        c2 = np.ones(y.shape[0])
        c = np.hstack((c1, c2))

        # create A
        I = np.eye(y.shape[0])
        A1 = np.hstack((-np.transpose(phi), -I))
        A2 = np.hstack((np.transpose(phi), -I))
        A = np.vstack((A1, A2))
        A = np.transpose(A)

        # create b
        b = np.hstack((-y, y))

        c = cvxopt.matrix(c.tolist())
        A = cvxopt.matrix(A.tolist())
        b = cvxopt.matrix(b.tolist())
        
        r = cvxopt.solvers.lp(c, A, b)
        self._param = np.array(list(r['x']))[:self.param_num]

        return self._param

    def objective_loss(self, x, y):
        prediction = self.predict(x)
        return np.mean(y - prediction)


class BayesianRegression(Algorithm):
    def __init__(self, transformation, param_num, name='', alpha=1, sigma=1):
        name = name + '_' + str(alpha) + '_' + str(sigma)
        super().__init__(transformation, param_num, name, algo_type=ALGO_TYPE.BR)
        self.alpha = alpha
        self.sigma = sigma

    def regress(self, x, y):
        y, phi = super().regress(x, y)

        # calculate sigma_param
        tmp1 = 1 / self.alpha * np.eye(self.param_num)
        tmp2 = 1 / self.sigma * np.dot(phi, np.transpose(phi))
        sigma_param = np.linalg.inv((tmp1 + tmp2))

        # calculate mu_param
        mu_param = 1 / self.sigma * np.dot(np.dot(sigma_param, phi), y)

        self._param = [mu_param, sigma_param]
        return self._param

    def predict(self, x):
        phi = self.transformation(x, self.param_num)
        mu = np.dot(phi, self.param[0])
        sigma = np.dot(np.dot(phi, self.param[1]), np.transpose(phi))

        return [mu, sigma]

    def mse_loss(self, x, y):
        mu, sigma = self.predict(x)
        return np.mean((y - mu) ** 2)

    def objective_loss(self, x, y):
        super().objective_loss(x, y)
        