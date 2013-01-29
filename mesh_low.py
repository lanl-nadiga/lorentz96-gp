import os
import itertools
import numpy
from numpy.fft import fft
import model
import sklearn.decomposition as sk
from multiprocessing import Pool

npar = 2
count = 8

try:
	rngfac
except NameError:
	rngfac = 0

coupling = numpy.linspace((1.-rngfac)*model.coupling.parents['lower'], (1.+rngfac)*model.coupling.parents['upper'], count)

forcing = numpy.linspace((1.-rngfac)*model.forcing.parents['lower'], (1.+rngfac)*model.forcing.parents['upper'], count)

amplitude = model.amplitude.value
timescale = model.timescale.value


input = numpy.array([ [coupling[i], forcing[l]]
			   for l in range(len(forcing))
			   for i in range(len(coupling))])

print 'mesh_low: input.shape is ', input.shape

indices = range(count**npar)

output = False

filename = 'output-mesh-low-{0}.npy'.format(rngfac)
if os.path.exists(filename):
	output = numpy.load(filename).reshape(count**npar, model.J)
else:
	def integrate(i):
		return model.integrate(10, *input[i])

	p = Pool(12)
	output = p.map(integrate, indices)
	numpy.save(filename, output)


def distance(Y, p = 2):
	return numpy.array([ sum( (Y - output[k]) ** p ) ** (1./p) for k in indices ])

indices = numpy.array(indices)

PCA = sk.PCA(0.99)
pca_output = PCA.fit_transform(numpy.array(output).reshape(count**npar, model.J))

PCA.n_components = 1
pca_output = pca_output[:,:PCA.n_components]

print 'mesh_low PCA components: ', PCA.n_components, PCA.explained_variance_ratio_

# Parameters and low_pca_values
d1 = numpy.array([ list(input[i]) + list(pca_output[i]) for i in range(count**npar) ])

