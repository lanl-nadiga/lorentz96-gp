import numpy
import pymc
from model import *

def final(P):
	print P
	result = numpy.array([sum((Y0 - interpolate_slow(Jlow, integrate_low(*p)))**2)**(1.0/2.0) for p in P])
	print result
	return result

#Y = pymc.Normal('Y', mu=final, value=0, observed=True, tau=1.)
