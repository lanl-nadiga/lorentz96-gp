'''
2 scale Lorenz 96
Comparing prm to high_res map with (prm,low_res) to high_res map
prm to high_res map:
  name = 'prm'
(prm,low_res) to high_res map
  name = 'id'
'''

# global rngfac, hi_thin

# rngfac = 0
# hi_thin = 9

import gpsm
import multiprocessing
import pymc
import sys

def gps(name):
	if name == "fft":
		return gpsm.gps_fft
	elif name == "id":
		return gpsm.gps
	elif name == "prm":
		return gpsm.gps_prm

def sample(name, index):

	gp = gps(name)[index]
	mc = pymc.MCMC(gp, db='pickle', dbname='gpsm-b0-{0}-{1}'.format(name, index))
	mc.sample(400, burn_till_tuned=True,thin=40)
	# mc.db.close()

def samplers(name):
	return [pymc.MCMC(gp, db=pymc.database.pickle.load('gpsm-b0-{0}-{1}'.format(name, gp['output_index']))) for gp in gps(name)]

	
def id1():
	'''
	NOTE: Even though in a function defintion, because of scoping rules, Run this at main level
	'''
	name = 'prm'
	index = 1
	
	gp = gps(name)[index]
	mc0 = pymc.MCMC(gp, db=pymc.database.pickle.load('gpsm-b0-{0}-{1}'.format(name,index)))
	tr1 = mc0.GP.f.trace()

	if name == 'prm':
		d1 = gpsm.mesh_low.d1[:,:4] # First 4 components of d1 are parameters
	elif name == "id":
		d1 = gpsm.mesh_low.d1 # After the 4 parameters are low_res PCA amplitudes


	# d2 = d1
	
	# reord = []
	# arng10 = arange(10)
	# for j in range(10):
	# 	if (j%2 == 0):
	# 		x = 10*j + arng10
	# 	else:
	# 		x = 10*(j+1) - 1 - arng10
	# 	reord = append(reord,x)
	# 	print j, reord
	# reord = reord.astype('int')

	# d1 = d2[reord]
		
	clf()
	plot(tr1[0](d1),'o',ms=5,label='Realizations of Learnt Map')
	[plot(tr1[i](d1),'o',ms=4) for i in range(0,len(tr1),1)]
	obs_loc = gpsm.mesh_high.indices
	obs = gpsm.mesh_high.pca_output[:,1]
	plot(obs_loc,obs,'r*',ms=20,label='Training Set')

	hi_thin = 6
	execfile('mesh_high.py')
	
	wheld_loc = indices
	wheld = pca_output[:,1]
	plot(wheld_loc,wheld,'yo',ms=10,label='Held-Out Set')

	leg=legend(loc=3)

	ylabel('Amplitude of high-resolution PCA #2')
	ylim(-8,6)
	
	if name == 'id':
		xlabel('Index (parameters; low-resolution PCA amplitudes)')
	elif name == "prm":
		xlabel('Index (parameters only)')

	savefig('{0}-{1}.png'.format(name,index+1))
	
	figure()
	plot(d1[indices][:,4],pca_output[:,1],'o')
	figure()
	plot(d1[indices][:,5],pca_output[:,1],'o')
	


if __name__ == "__main__":
	name = sys.argv[1]
	if len(sys.argv) == 2:
		for i in range(len(gps(name))):
			print i
		sys.exit(0)

	index = int(sys.argv[2])
	sampler = sample(name, index)
	sys.exit(0)

