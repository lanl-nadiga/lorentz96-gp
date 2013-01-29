import numpy
from numpy import *
from scipy.integrate import ode
import pymc
import os

T = 1
dt = T
Nt = int(T / dt)

J = 40

def dX_dt_old(h, b, c, F, K):

    if h<0: F=-abs(F)
    def dxdt(X):
	J = len(X) / (1+K)

	in1  = empty(J+4) # 4 ghost points

	in1[2:-2] = X[:J]
	in1[0] = X[J-2]
	in1[1] = X[J-1]
	in1[-2] = X[0]
	in1[-1] = X[1]

	in2  = empty(J*K+4) # 4 ghost points

	in2[2:-2] = X[J:]
	in2[0] = X[-2]
	in2[1] = X[-1]
	in2[-2] = X[J]
	in2[-1] = X[J+1]

	rhs = empty(J*(1+K))

	rhs[:J] =       in1[1:-3] * ( in1[3:-1] - in1[:-4] ) - in1[2:-2] + F
	rhs[J:] = c*( b*in2[3:-1] * ( in2[1:-3] - in2[4:]  ) - in2[2:-2])
	
	for j in range(J):
	    kbeg = J + K*j
	    rhs[j] -= h*c/b*sum(X[kbeg:kbeg+K])
	    x1 = h*c/b*X[j]
	    rhs[kbeg:kbeg+K] += x1
	    
	return rhs

    return lambda t, X: dxdt(X)

def dX_dt(h, b, c, F, K):
    
    if h<0: F=-abs(F)
    
    def dxdt(X):
	J = len(X) / (1+K)
	hc_ovr_b = h*c/b
	
	slow = X[:J]
	slow_2d = repeat(slow,K)
	fast = X[J:]
	fast_2d = fast.reshape(J,K)

	in1 = numpy.hstack((slow[-2:],slow,slow[:2]))# 4 ghost points
	in2 = numpy.hstack((fast[-2:],fast,fast[:2]))# 4 ghost points

	rhs = empty(J*(1+K))

	rhs_slow = rhs[:J]
	rhs_fast = rhs[J:]

	rhs_fast[:] = c* ( b*in2[3:-1] * ( in2[1:-3] - in2[4:]  ) - fast) + hc_ovr_b * slow_2d

	rhs_slow[:] =        in1[1:-3] * ( in1[3:-1] - in1[:-4] ) - slow + F \
		      - hc_ovr_b*sum(fast_2d,axis=1)


	# for j in range(J):
	#     rhs_fast_2d[j,:] += hc_ovr_b * slow[j] #THIS IS SLOW

	return rhs

    return lambda t, X: dxdt(X)
	
def traj(J, K, T_spinup, T, dt, h=1, F=10, b=10, c=10):

    ode1 = ode(dX_dt(h, b, c, F, K))
    ode1.set_integrator('dopri5', rtol=1e-3, atol=1e-6, nsteps=10000)

    X = sin(arange(J*(1+K)))
    # X = random.random_sample(J*(1+K)) # 0 to 1
    ode1.set_initial_value(X, 0)

    # Xhist = X[:J]
    ode1.integrate(ode1.t+T_spinup)
    Xhist = ode1.y[:J]

    T += T_spinup
    
    while ode1.successful() and ode1.t < T:
	ode1.integrate(ode1.t+dt)
	Xhist = vstack((Xhist,ode1.y[:J]))

	# print 'pid={0}	h={2}	b={3}	c={4}	F={5}	t={1}'.format(os.getpid(), ode1.t, h, b, c, F), Xhist[-1].min(),Xhist[-1].max()

    return Xhist


def integrate(K, h=1, F=10, b=10, c=10):

    integrator = ode(dX_dt(h, b, c, F, K))
    integrator.set_integrator('dopri5', rtol=1e-3, atol=1e-6, nsteps=10000)

    X = random.random_sample(J*(1+K))
    integrator.set_initial_value(X, 0)

    # Xhist = X[0:J]
    for it in range(1,1+Nt):
	integrator.integrate(it*dt)
	# Xhist = append(Xhist,integrator.y[0:J])

	print 'pid={0}	h={2}	b={3}	c={4}	F={5}	t={1}'.format(os.getpid(), integrator.t, h, b, c, F)

	# Xhist.shape = (1+Nt,J)
	# return Xhist

    return integrator.y[0:J]

nprms = 2

coupling = pymc.Uniform('h', lower=1, upper=4, value=4)
forcing = pymc.Uniform('F', lower=5, upper=15, value=10)
amplitude = pymc.Uniform('b', lower=10, upper=10, value=10)
timescale = pymc.Uniform('c', lower=10, upper=10, value=10)


# print dX_dt_old(1,10,10,10,2)(0,sin(arange(6)))
# print dX_dt(1,10,10,10,2)(0,sin(arange(6)))

foo = traj(40,10,30,100,0.5,h=-3.25)
