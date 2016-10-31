import csv
import cv2
import sys
import numpy as np
from scipy import sparse
from scipy import optimize
import fileinput
import re
import math

def norm_pdf(x, loc, std_dev):
	return math.exp(-(float(x) - loc)**2/(2*std_dev))/math.sqrt(2*math.pi)/std_dev
	
if __name__ == '__main__':
	F = 44100
	f0 = 21.675
	r = 0.109
	
	# if len(sys.argv) > 1:
	# 	with open('real_im_here_01_23_00.csv', 'r') as f:
	# 		reader = csv.reader(f, delimiter=',')
			
	# 		# recall fft_freqs: [0, 1, 2, ..., floor(n-1)/2, -floor(n/2), ..., -2, -1] / (d * n)
	# 		# => there are n elements
	# 		weights = np.array(reader.next(), dtype=np.float32)
	# else:
	outfile = sys.argv.pop()
	weights = np.array(re.split(',?\s*', " ".join([a.strip() for a in fileinput.input()])), dtype=np.float32)
	
	N = len(weights)
	freq_to_idx = lambda f: int(math.floor(float(f) * 1471 / F))
	idx_to_freq = lambda idx: float(idx) * F / 1471
	freq_to_F_domain = lambda f: np.log(float(f) / f0) / r
	F_domain_to_freq = lambda f_d: r * np.exp(float(f_d) * f0)
	
	translated_range = (0, N)

	F_std_dev = math.sqrt(3/(math.log(25.0/16)))
	# let's go 5 std deviations away
	data = [1.0/math.sqrt(2*math.pi)/F_std_dev]
	print(translated_range)
	coords = [[0], [0]]
	for i in range(translated_range[0], translated_range[1]):
		freq_i = idx_to_freq(i)
		R = (max(translated_range[0], freq_to_idx(freq_i / math.exp(r * 5 * F_std_dev))), min(translated_range[1], freq_to_idx(freq_i * math.exp(r * 5 * F_std_dev))))
		coords[0] += [i] * (R[1] - R[0])
		coords[1] += range(R[0], R[1])
		data += [norm_pdf(freq_to_F_domain(idx_to_freq(idx)), freq_to_F_domain(idx_to_freq(i)), F_std_dev)/(R[1]-R[0]) for idx in range(R[0], R[1])]

	# np.savetxt('%s.csv' % outfile, data, delimiter=',')
	# cv2.imwrite('%s.png' % outfile, sparse.csr_matrix((data, coords), dtype=np.float32).multiply(2000).toarray()) # to 255
	np.savetxt('%s.csv' % outfile, optimize.nnls(sparse.csr_matrix((data, coords), dtype=np.float32).toarray(), weights[0:translated_range[1]])[0], delimiter=',')