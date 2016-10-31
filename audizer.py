import sys
import math
# import numpy as np
import cv2
import audioop
import wave
import numpy as np
# from matplotlib import pyplot
from functools import reduce
from operator import mul

class DisjointPoint:
	def __init__(self, rep = None, rank = 0):
		self.rep = rep
		self.rank = rank

class DisjointSet:
	def __init__(self):
		self.d = {}
	
	def find(self, k):
		if self.d[k].rep==k:
			return k
		else:
			self.d[k].rep = self.find(k)
			return self.d[k].rep
	
	def union(self, i, j):
		a = self.d[self.find(i)]
		b = self.d[self.find(j)]
		if a.rank>b.rank:
			b.rep = a
			try:
				a.update_aggregates(b)
			except AttributeError:
				pass
		else:
			a.rep = b
			try:
				b.update_aggregates(a)
			except AttributeError:
				pass
			
			if b.rank == a.rank:
				b.rank += 1

class Point(DisjointPoint):
	def __init__(self, x, y, label):
		super().__init__()
		self.x = x
		self.y = y
		self.label = label
		self.bounds = { 'x': [None, None], 'y': [None, None] }
		self.count = 1
		
	def update_aggregates(incoming):
		self.bounds['x'][0] = min(self.bounds['x'][0], incoming.x)
		self.bounds['x'][1] = max(self.bounds['x'][1], incoming.x)
		self.bounds['y'][0] = min(self.bounds['y'][0], incoming.y)
		self.bounds['y'][1] = max(self.bounds['y'][1], incoming.y)
		self.count += incoming.count

def poly_area(P):
	x = P[0:,0]
	y = P[0:,1]
	return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def dominant_color(frame, xlim, ylim):
	# divide and conquer averaging algorithm to prevent overflow errors
	sum_weight = 0.0
	weighted_sum = np.array([0, 0, 0], dtype=np.float32)
	if xlim[0] == xlim[1] or ylim[0] == ylim[1]:
		print(xlim, ylim)
		flat = frame.reshape((reduce(mul, frame.shape)/frame.shape[-1], frame.shape[-1]))
		for pixel in flat:
			weight = pixel[1] * pixel[2]
			sum_weight += weight
			weighted_sum += weight * pixel
		return weighted_sum / sum_weight
	else:
		xavg = xlim[0] + (xlim[1]-xlim[0])//2
		yavg = ylim[0] + (ylim[1]-ylim[0])//2
		flat = [	dominant_color(frame, (xlim[0], xavg), (ylim[0], yavg)), dominant_color(frame, (xavg+1, xlim[1]), (yavg, ylim[1])), dominant_color(frame, (xlim[0], xavg), (yavg+1, ylim[1])), dominant_color(frame, (xavg+1, xlim[1]), (yavg+1, ylim[1])) ]
		for pixel in flat:
			weight = pixel[1] * pixel[2]
			sum_weight += weight
			weighted_sum += weight * pixel
		return weighted_sum / sum_weight

if __name__ == '__main__':
	W = 518
	N = 61
	left = 65
	bottom = 181
	bar_width = 4
	F = 44100 # uncompressed WAV sampling rate of 44.1kHz

	freqs = [23.336 * math.exp(0.1203*x) for x in range(N)]
	V = cv2.VideoCapture(sys.argv[1])
	
	audio = []
	video_frame_count = 0
	audio_video_frame_desync = 1/V.get(cv2.cv.CV_CAP_PROP_FPS) % 1/44100
	audio_video_desync = 1.0 # at each iteration, add (1/V.get(cv2.cv.CV_CAP_PROP_FPS) % 1/44100) mod 1/44100
	audio_frame_baseline = int(F/V.get(cv2.cv.CV_CAP_PROP_FPS)) - 1
	audio_frames_per_video_frame = F/V.get(cv2.cv.CV_CAP_PROP_FPS)
	for j in range(int(sys.argv[2])):
		V.grab() # skip first few frames
	color = dominant_color(V.read()[1][0:bottom,left:left+W]) # process dominant color of bars from first frame
	while V.grab():
		raw_v = V.retrieve()[1]
		v = cv2.cvtColor(raw_v, cv2.COLOR_RGB2HSV)
		for i in range(61):
			# transform to vectors in XYSV space
			M = [np.array([x, y, v[x][y][1], v[x][y][2]], dtype=np.uint16) for y in range(0, bottom) for x in range(left, left + bar_width) ]
			criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
			compactness, labels, centers = cv2.kmeans(np.array(M, dtype=np.float32), 2, criteria, 2, cv2.KMEANS_RANDOM_CENTERS)
			bar_cluster_idx, bar_cluster_center = reduce(lambda prev, next: prev if prev[1] > np.linalg.norm((next[1][2], next[1][3])) else (next[0], np.linalg.norm((next[1][2], next[1][3]))), enumerate(centers), (None, float('-inf'))) # most saturated and highest-value cluster (might not work the best on Electronic/unclassified songs)
			bar_cluster = np.reshape((labels == bar_cluster_idx).astype(np.uint8), (bottom, bar_width))
			# cv2.imwrite('temp.png', raw_v[0:bottom,left:left+bar_width])
			cluster_contours, hierarchy = cv2.findContours(bar_cluster, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			
			max_area = float('-inf')
			max_cluster = None
			for cluster in cluster_contours:
				A = poly_area(cluster.reshape(cluster.shape[0], cluster.shape[2]))
				if A > max_area:
					max_area = A
					max_cluster = cluster
			
			extremum = reduce(cluster, lambda p, n: n if n[1] < p[1] else p, (None, float('inf')))
			for j in range(audio_frame_baseline + int(audio_video_desync)):
				audio_frame = int(video_frame_count * audio_frames_per_video_frame) + 1 + j
				audio[audio_frame] = int(math.cos(j * 2 * math.PI * freqs[j] / F) * np.iinfo(np.int16).max * (bottom - extremum) / bottom)
			audio_video_desync = (audio_video_desync + audio_video_frame_desync) - int(audio_video_desync)
		video_frame_count += 1
	wave.open(sys.argv[3], 'w+').writeframesraw(audio)
			# Dlabels = [Point(pos % bar_width, pos // bar_width, label) for pos, label in enumerate(labels)]
			# for pos, point in enumerate(Dlabels):
			# 	directions = [pos - 1, pos + 1, pos - bar_width, pos + bar_width]
			# 	for d in directions:
			# 		if d > 0 and d < bar_width * bottom and Dlabels[d].label == point.label:
			# 			point.union(Dlabels[d])