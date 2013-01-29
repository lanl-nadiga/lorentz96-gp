import pymc
from pymc import gp as gp
import numpy as np
import stochastic
import model
import mesh_low
import mesh_high

def mean(I, h, b, c, F):
	return np.zeros(len(I))
	# return np.ones(len(I))*(F-10)*5

def gp_model(d1, output, diff_deg_val=None, amp_val=None, scale_val=None):

	# Prior parameters of C
	diff_deg = pymc.Uniform('diff_deg', 1., 3.)
	amp = pymc.Lognormal('amp', mu=.4, tau=.1)#, value=1)
	scale = pymc.Lognormal('scale', mu=.5, tau=1)#, value=1)

	if diff_deg_val is not None: diff_deg.value = diff_deg_val
	if amp_val is not None: amp.value = amp_val
	if scale_val is not None: scale.value = scale_val

	
	# The covariance dtrm C is valued as a Covariance object.
	@pymc.deterministic
	def C(eval_fun = gp.matern.euclidean, diff_degree = diff_deg, amp = amp, scale = scale):
		return gp.NearlyFullRankCovariance(eval_fun,
						   diff_degree = diff_degree, amp = amp, scale = scale)
	# The mean M is valued as a Mean object.
	@pymc.deterministic
	def M(eval_fun = mean, h=model.coupling, b=model.amplitude, c=model.timescale, F=model.forcing):
		return gp.Mean(eval_fun, h=h, b=b, c=c, F=F)

	# Define, on appropriate mesh, the GP with mean M and covariance C
	GP = gp.GPSubmodel('gpsm', M, C, d1)

	# Observations for the map to learn
	data = pymc.Normal('data', mu=GP.f(d1),
			   tau = 1e4, value=output, observed=True)
	
	return locals()

def sample2(tr_in, tr_out,hypers):
	mc = pymc.MCMC(gp_model(tr_in, tr_out))#,*hypers))
	mc.sample(100, burn = 500, thin=1)

	return mc

def like_pair_1map(input):

	pair = input[0]
	p2h_tr_full = input[1][0]

	try:
		hypers = input[1][1:]
	except IndexError:
		hypers = [None]*3
		
	#Make copy and move pair to end
	tr_set = p2h_tr_full.copy()
	tr_set[[-1,pair[0]]] = tr_set[[pair[0],-1]]
	tr_set[[-2,pair[1]]] = tr_set[[pair[1],-2]]

	mc2 = sample2(tr_set[:-2,:-1], tr_set[:-2,-1], hypers) #leave out last 2 points
	ntr = len(mc2.GP.f.trace())
	est2 = np.array([mc2.GP.f.trace()[itr]([tr_set[-1,:-1]])   #value at heldout pt. without extra point
		      for itr in range(ntr)]).mean()

	mc1 = sample2(tr_set[:-1,:-1], tr_set[:-1,-1], hypers) #leave out only heldout pt
	est1 = np.array([mc1.GP.f.trace()[itr]([tr_set[-1,:-1]])   #value at heldout pt. with extra point
		      for itr in range(ntr)]).mean()
	
	return abs(est2 - tr_set[-1,-1]) - abs(est1 - tr_set[-1,-1]) 


def unlike_pair(input):

	pair = input[0]
	p2l_full = input[1][0]
	l2h_full = input[1][1]

	try:
		hypers = input[1][2:]
	except IndexError:
		hypers = [None]*3
		
	#Make copy and move pair to end
	p2l_set = p2l_full.copy()
	p2l_set[[-1,pair[0]]] = p2l_set[[pair[0],-1]] #testpt.
	p2l_set = np.delete(p2l_set, pair[1], axis=0) # delete holdout l2h point from p2l

	l2h_set = l2h_full.copy()
	l2h_set[[-1,pair[1]]] = l2h_set[[pair[1],-1]] #holdout

	mc2 = sample2(p2l_set[:-1,:-1], p2l_set[:-1,-1], hypers) #leave out last point
	ntr = len(mc2.GP.f.trace())

	tmp2 = np.array([mc2.GP.f.trace()[itr]([l2h_set[-1,:2]]) #value of low at hires heldout pt.
		      for itr in range(ntr)]).mean()

	mc1 = sample2(p2l_set[:,:-1], p2l_set[:,-1], hypers) 
	tmp1 = np.array([mc1.GP.f.trace()[itr]([l2h_set[-1,:2]])  #value of low at hires heldout pt.
		      for itr in range(ntr)]).mean()
	
	mc_l2h = sample2(l2h_set[:-1,-2], l2h_set[:-1,-1], hypers) #leave out last point; dont use prms
	ntr = len(mc_l2h.GP.f.trace())
	
	est2 = np.array([mc_l2h.GP.f.trace()[itr]([tmp2]) #value at hires heldout pt. without extra point
		      for itr in range(ntr)]).mean()

	est1 = np.array([mc_l2h.GP.f.trace()[itr]([tmp1]) #value at hires heldout pt. with extra point
		      for itr in range(ntr)]).mean()
	# print tmp2,tmp1,l2h_set[-1,-2], est2,est1,l2h_set[-1,-1]
	return abs(est2 - l2h_set[-1,-1]) - abs(est1 - l2h_set[-1,-1]) 


def like_pair(input):

	pair = input[0]
	p2l_full = input[1][0]
	l2h_full = input[1][1]

	try:
		hypers = input[1][2:]
	except IndexError:
		hypers = [None]*3
		
	p2l_set = p2l_full.copy()
	p2l_set = np.delete(p2l_set, pair[1], axis=0) # delete heldout l2h point from p2l

	#Make copy and move pair to end
	l2h_set = l2h_full.copy()
	l2h_set[[-1,pair[1]]] = l2h_set[[pair[1],-1]] #holdout
	l2h_set[[-2,pair[0]]] = l2h_set[[pair[0],-2]] #test point

	mc_p2l = sample2(p2l_set[:-1,:-1], p2l_set[:-1,-1], hypers) #leave out last point
	ntr = len(mc_p2l.GP.f.trace())

	tmp = np.array([mc_p2l.GP.f.trace()[itr]([l2h_set[-1,:2]]) #value of low at hires heldout pt.
		      for itr in range(ntr)]).mean()

	mc_l2h_1 = sample2(l2h_set[:-1,-2], l2h_set[:-1,-1], hypers) #Use testpt.; leave out heldout pt; dont use prms
	ntr = len(mc_l2h_1.GP.f.trace())
	
	est1 = np.array([mc_l2h_1.GP.f.trace()[itr]([tmp]) #value at heldout pt. without test point
		      for itr in range(ntr)]).mean()

	mc_l2h_2 = sample2(l2h_set[:-2,-2], l2h_set[:-2,-1], hypers) #leave out testpt and heldout pt; dont use prms
	est2 = np.array([mc_l2h_2.GP.f.trace()[itr]([tmp]) #value at heldout pt. without test point
		      for itr in range(ntr)]).mean()

	# print tmp2,tmp1,l2h_set[-1,-2], est2,est1,l2h_set[-1,-1]
	return abs(est2 - l2h_set[-1,-1]) - abs(est1 - l2h_set[-1,-1]) 
