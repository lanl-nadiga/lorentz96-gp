import pymc
from pymc import gp as gp
import numpy
import stochastic
import model
import mesh_low
import mesh_high

def mean(I, h, b, c, F):
	return numpy.zeros(len(I))
	# return numpy.ones(len(I))*(F-10)*5

def gp_model(parameter_lattice_points, d1, output_high, output_index):
#	nu = pymc.Normal('nu', 1.5, 1.5, value=1.5)
#	phi = pymc.Lognormal('phi', mu=.4, tau=1, value=1)
#	theta = pymc.Lognormal('theta', mu=.5, tau=1, value=1)
	#C = gp.Covariance(eval_fun = gp.matern.euclidean, diff_degree = nu, amp = phi, scale = theta)
	#M = gp.Mean(eval_fun = stochastic.final, h=model.coupling, b=model.amplitude, c=model.timescale, F=model.forcing)

	# Prior parameters of C

	nu = pymc.Uniform('nu', 1., 3.)
	phi = pymc.Lognormal('phi', mu=.4, tau=.1)#, value=1)
	theta = pymc.Lognormal('theta', mu=.5, tau=1)#, value=1)
	
	# The covariance dtrm C is valued as a Covariance object.

	@pymc.deterministic
	def C(eval_fun = gp.matern.euclidean, diff_degree = nu, amp = phi, scale = theta):
		return gp.NearlyFullRankCovariance(eval_fun, diff_degree = diff_degree, amp = amp, scale = scale)

#	beta = numpy.empty(d1.shape[1], dtype=object)
#	for i in range(d1.shape[1]):
#		beta[i] = pymc.Normal('beta_{0}'.format(i), mu=1, tau=1)

#	h_weight = pymc.Normal('h_weight', mu=1., tau=0.5, value=1.)
#	b_weight = pymc.Normal('b_weight', mu=1., tau=0.5, value=1.)
#	c_weight = pymc.Normal('c_weight', mu=1., tau=0.5, value=1.)
#	F_weight = pymc.Normal('F_weight', mu=1., tau=0.5, value=1.)

	# The covariance dtrm C is valued as a Covariance object.
	#@pymc.deterministic
	#def C(eval_fun = gp.matern.euclidean, diff_degree=nu, amp=phi, scale=theta):
	#	return gp.NearlyFullRankCovariance(eval_fun, diff_degree=diff_degree, amp=amp, scale=scale)


#		def square_difference(a, b):
#			return a**2 + b**2 - 2*a*b
#		return numpy.array([ sum( square_difference(numpy.array(p), numpy.array([h,b,c,F])) * numpy.array([h_weight, b_weight, c_weight, F_weight]) ) for p in P])

# The mean M is valued as a Mean object.

	@pymc.deterministic
	def M(eval_fun = mean, h=model.coupling, b=model.amplitude, c=model.timescale, F=model.forcing):
		return gp.Mean(eval_fun, h=h, b=b, c=c, F=F)

	# C = gp.Covariance(eval_fun = gp.matern.euclidean, diff_degree = 1.4, amp = 1., scale = 1.)
	
	#C = gp.Covariance(eval_fun = gp.matern.euclidean, diff_degree = nu, amp = phi, scale = theta)
	#M = gp.Mean(eval_fun = test_mean, h=model.coupling, b=model.amplitude, c=model.timescale, F=model.forcing, h_weight=h_weight, b_weight=b_weight, c_weight=c_weight, F_weight=F_weight)

	# Observations for the map from mesh_low to mesh_high are located at d1[parameter_lattice_points])
	
	GP = gp.GPSubmodel('gpsm', M, C, d1[parameter_lattice_points])
	#gp.observe(M, C, mesh_low.input.reshape(625, 4), mesh_low.distance(model.Y0), numpy.zeros(625))
	#C.observe(mesh_low.input.reshape(625, 4), numpy.ones(625))
	#M.observe(C, mesh_low.input.reshape(625, 4), mesh_low.distance(model.Y0))

	# Observation variance

	variance = pymc.Lognormal('variance', mu=-1, tau=1., value=0.0001)
	data = pymc.Normal('data', mu=GP.f(d1[parameter_lattice_points]), tau = 1./variance, value=output_high[:, output_index], observed=True)

	l = locals()
	return { k: l[k] for k in ['parameter_lattice_points', 'output_index', 'nu', 'phi', 'theta', 'GP', 'variance', 'data'] }

# learn map from (params,low_pca_values) to high_pca_values
gps = [gp_model(mesh_high.indices, mesh_low.d1, mesh_high.pca_output, i) for i in range(mesh_high.PCA.n_components)]

# learn map from params to high_pca_values
gps_prm = [gp_model(mesh_high.indices, mesh_low.d1[:,:4], mesh_high.pca_output, i) for i in range(mesh_high.PCA.n_components)]

# learn map from params to low pca values
gps_prm_to_low = [gp_model(mesh_low.indices, mesh_low.d1[:,:2], mesh_low.pca_output, i) for i in range(mesh_low.PCA.n_components)]

# gps_fft = [ gp_model(mesh_low.d1_fft, mesh_high.pca_fft_output, i) for i in range(mesh_high.PCA_FFT.n_components) ]
