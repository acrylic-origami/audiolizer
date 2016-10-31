import numpy as np
import sys
import audiotools
import cv2
import warnings
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader as vid

def poly_area(P):
	x = P[0:,0]
	y = P[0:,1]
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
	
if __name__ == '__main__':
	W = 518
	N = 62
	left = 67
	bottom = 181
	top = 56
	bar_width = 4
	vid_offset = 0.021 # video is 21ms ahead of audio
	advancer = 0
		
	V = vid(sys.argv[1])
	
	A = audiotools.open(sys.argv[2])
	F = A.sample_rate()
	audio_limits = (float(sys.argv[3]) * F, (float(sys.argv[5]) * F if len(sys.argv) > 4 else float(A.total_frames())))
	
	P = A.to_pcm()
	Q = P.read(int(audio_limits[0] + vid_offset * V.fps + audio_limits[1] - audio_limits[0]))
	
	# create extension
	V.initialize(V.duration/2)
	with warnings.catch_warnings():
		warnings.filterwarnings('error', 'oooh', UserWarning)
		
		v = V.read_frame()
		hsv_v = cv2.cvtColor(v, cv2.COLOR_RGB2HSV)
		hsv_frame = hsv_v[0:bottom, left:left+W]
		
		# calculate probable bar color
		weights = np.apply_along_axis(lambda x: (x[1]*x[2],), -1, hsv_frame.astype(np.uint16)) # weight by saturation and value, uint16 suffices for max(uint8*uint8) = 2^16
		positional_weights = np.reciprocal(np.arange(weights.shape[0], 0, -1).astype(np.float32)).reshape([weights.shape[0]]+(len(weights.shape)-1)*[1])
		reweighted = np.multiply(weights, positional_weights)
		reweighted_frame = np.multiply(reweighted, hsv_frame)
		reweighted_sum = np.sum(reweighted)
		while len(reweighted_frame.shape) > 1:
			reweighted_frame = np.sum(reweighted_frame, -2)
		bar_color = (reweighted_frame / reweighted_sum).astype(np.uint8)
		
	extension = np.zeros((40,) + hsv_frame.shape[1:], dtype=np.uint8)
	for i in range(N):
		cv2.rectangle(extension, (int(i * float(W)/N), 0), (int(i * (float(W)/N) + bar_width), 39), [int(color) for color in bar_color], cv2.cv.CV_FILLED)
	
	# correlate
	V.initialize(float(sys.argv[3]))
	with open(sys.argv[4] + '.csv', 'w+') as f:
		f.write('bar, freq')
		for j in range(int((audio_limits[1] - audio_limits[0]) / (float(F) / V.fps))):
			# print(j)
			bs = [Q.frame(int(audio_limits[0] + j * (float(F) / V.fps) + i + vid_offset * V.fps)).to_bytes(False, True)[0:2] for i in range(int(F/V.fps))]
			L = np.fromstring("".join(bs), dtype=np.int16)
			
			v = V.read_frame()
			hsv_v = cv2.cvtColor(v, cv2.COLOR_RGB2HSV)
			hsv_frame = np.vstack((hsv_v[0:bottom, left:left+W], extension))
			all_contours = []
			extrema = []
			for i in range(N):
				M = [np.array([x, y, hsv_frame[y][x][1], hsv_frame[y][x][2] ** 2], dtype=np.uint16) for y in range(0, hsv_frame.shape[0]) for x in range(int(i * float(W)/N), int(i * (float(W)/N) + bar_width)) ]
				criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
				compactness, labels, C = cv2.kmeans(np.array(M, dtype=np.float32), 2, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)
				bar_cluster_idx, bar_cluster_center = reduce(lambda prev, next: prev if prev[1] > next[1][2] * next[1][3] else (next[0], next[1][2] * next[1][3]), enumerate(C), (None, float('-inf')))
				bar_cluster = np.reshape((labels == bar_cluster_idx).astype(np.uint8), (hsv_frame.shape[0], bar_width))
				cluster_contours, hierarchy = cv2.findContours(bar_cluster, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				max_area = float('-inf')
				max_cluster = None
				if(len(cluster_contours) > 0):
					for idx, cluster in enumerate(cluster_contours):
						A = poly_area(cluster.reshape(cluster.shape[0], cluster.shape[2]))
						if A > max_area:
							max_area = A
							max_cluster = idx
						cluster_contours[idx] = np.add(cluster, np.array([int(i * float(W)/N), 0]))
					
					extrema.append(reduce(lambda p, n: n if n < p else p, cluster_contours[max_cluster][:,:,1].reshape(cluster_contours[max_cluster].shape[0]), float('inf')))
				else:
					extrema.append(bottom)
				
				all_contours.append(cluster_contours)
				# cv2.drawContours(hsv_frame, cluster_contours, max_cluster, (0, 255, 255), cv2.cv.CV_FILLED)
			heights = (bottom - np.array(extrema, dtype=np.float32)) / (bottom - top)
			advancer += np.argmax(heights[advancer:])
			l = heights[max(advancer-1, 0)]
			r = heights[min(advancer+1, N-1)]
			
			cv2.drawContours(hsv_frame, all_contours[advancer], max_cluster, (0, 255, 255), cv2.cv.CV_FILLED)
			bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
			cv2.imwrite('%s.%d.png' % (sys.argv[4], j), bgr_frame)
			
			print('%.2f\t%d\t%.5f\t%.5f\t%.5f' % (float(sys.argv[3]) + float(j) / V.fps, advancer, l, heights[advancer], r))
			f.write('%.5f,' % (advancer + (heights[advancer] - max(l, r)) / (heights[advancer] - min(l, r))))
			f.write('%.5f\n' % np.fft.fftfreq(int(F / V.fps), d=1.0/F)[10 + np.argmax(np.absolute(np.fft.fft(L))[10:15000//V.fps])]) # scan from ~300Hz up to ~15kHz
			
			