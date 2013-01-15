import model
import sampler
import sys, os
import numpy
import mesh_high, mesh_low

if len(sys.argv) != 1 + 4:
	print "Usage: {0} {1} <h> <b> <c> <F>"
	sys.exit(1)

P = [float(s) for s in sys.argv[1:]]
h, b, c, F = P

file_pattern="{0}-h={1}-b={2}-c={3}-F={4}.npy"

data = {}
for resolution in ["low", "high"]:
	file = file_pattern.format(resolution, h, b, c, F)
	if os.path.exists(file):
		data[resolution] = numpy.load(file)
		print "loaded data from {0}".format(file)
		print sum(data[resolution])
	else:
		data[resolution]= model.integrate(h, b, c, F, model.Jlow if resolution == "low" else model.Jhigh)
		numpy.save(file, data[resolution])

prediction = {}
prediction['high'] = mesh_high.PCA.inverse_transform(
	numpy.array(
		[
			mc.GP.f.value(
				[P + list(mesh_low.PCA.transform(data['low'][0:model.Jlow]))]
			) for mc in sampler.mcs 
		]
	).reshape(
		mesh_high.PCA.n_components
	)
)

print sum(abs(prediction['high'] - data['high']))/sum(abs(data['high']))


