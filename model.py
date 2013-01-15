import numpy
from numpy import *
from scipy.integrate import ode
import pymc
import os

T = 1
dt = T
Nt = int(T / dt)

J = 40

def dX_dt(h, b, c, F, K):

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
		rhs[J:] = c* (b*in2[1:-3] * ( in2[3:-1] - in2[:-4] ) - in2[2:-2])

		for j in range(J):
			x1 = h*c/b*X[j]
			rhs[j] -= x1
			kbeg = J + K*j
			rhs[kbeg:kbeg+K] += x1

		return rhs

	return lambda t, X: dxdt(X)
	
def integrate(h, b, c, F, K):

	integrator = ode(dX_dt(h, b, c, F, K))
	integrator.set_integrator('dopri5', nsteps=10000)

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

def integrate_high(h, b, c, F):
	return integrate(h, b, c, F, 20)
def integrate_low(h, b, c, F):
	return integrate(h, b, c, F, 4)

coupling = pymc.Uniform('h', lower=1, upper=4, value=4)
amplitude = pymc.Uniform('b', lower=10, upper=10, value=10)
timescale = pymc.Uniform('c', lower=10, upper=10, value=10)
forcing = pymc.Uniform('F', lower=5, upper=15, value=10)


