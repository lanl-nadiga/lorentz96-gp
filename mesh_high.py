import os
import itertools
import numpy
from numpy.fft import fft
import model
import sklearn.decomposition as sk
from multiprocessing import Pool
import mesh_low
from mesh_low import count, input

try:
	rngfac
except NameError:
	rngfac = 0

try:
	hi_thin
except NameError:
	hi_thin = 9

print 'mesh_high:  hi_thin is ', hi_thin

filename = 'indices-high-{0}-{1}.npy'.format(hi_thin,rngfac)
if os.path.exists(filename):
	indices = numpy.load(filename)
else:
	indices = numpy.array([k for k in range(len(mesh_low.indices)) if k % hi_thin == 0])
	numpy.save(filename, indices)

def integrate(i):
	print 'i = %d' % i
	print 'i_high = %s' % str(indices[i])
	print 'ii = %s' % str(mesh_low.indices[indices[i]])
	print 'input = %s' % str(mesh_low.input[mesh_low.indices[indices[i]]])
	return model.integrate_high(*mesh_low.input[mesh_low.indices[indices[i]]])

filename = 'output-mesh-high-{0}-{1}.npy'.format(hi_thin,rngfac)
if os.path.exists(filename):
	output = numpy.load(filename)
else:
	p = Pool(12)
	output = p.map(integrate, range(len(indices)))
	# output = map(integrate, range(len(indices)))
	numpy.save(filename, output)

def distance(Y, p = 2):
	print "warning: this is a bad function!"
	return numpy.array([ sum( (Y - model.interpolate_slow(model.J, output[k])) ** p) ** (1./p) for k in range(len(indices)) ])

#PCA = mlab.PCA(output[:,0:model.Jhigh])
#pca_output = PCA.project(output[:,0:model.Jhigh])
if hi_thin == 9:
	PCA = sk.PCA(0.99)
	pca_output = PCA.fit_transform(output)
else:
	PCA = gpsm.mesh_high.PCA
	pca_output = PCA.transform(output)
print 'mesh_high PCA components: ', PCA.n_components, PCA.explained_variance_ratio_

# fft_output = fft(output)
# PCA_FFT = sk.PCA(0.999)
# pca_fft_output =PCA_FFT.fit_transform(numpy.array([ [z.real, z.imag] for z in fft_output ]).reshape(63, 2*400))
