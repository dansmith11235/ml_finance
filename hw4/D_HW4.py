import numpy as np
#from cvxpy import *
import cvxpy as cvx
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'


np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)

L_vals = [1,2,4]

for L in L_vals:

    w = cvx.Variable(n)
    gamma = cvx.Parameter(nonneg=True)# gamma = Parameter(nonneg=True)
    ret = mu.T*w
    risk = cvx.quad_form(w, Sigma)
    prob = cvx.Problem(cvx.Maximize(ret - gamma*risk),[cvx.norm(w,1) <= L])


    SAMPLES = 100
    risk_data = np.zeros(SAMPLES)
    ret_data = np.zeros(SAMPLES)
    gamma_vals = np.logspace(-2, 3, num=SAMPLES)

    for i in range(SAMPLES):
        gamma.value = gamma_vals[i]
        prob.solve()
        risk_data[i] = cvx.sqrt(risk).value
        ret_data[i] = ret.value



    markers_on = [29, 40]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(risk_data, ret_data, 'g-')
    for i in range(n):
        plt.plot(cvx.sqrt(Sigma[i,i]).value, mu[i], 'ro')
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    plt.title("L = "+str(L))
plt.show()