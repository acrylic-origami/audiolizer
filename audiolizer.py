import sys
import math
import numpy as np
from scipy import interpolate
import cv2
import warnings
import wave
import struct
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader as vid
from functools import reduce
import cmath
import random

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
	F = 44100 # uncompressed WAV sampling rate of 44.1kHz
	
	freqs = [21.675 * math.exp(0.109*x) for x in range(N)]
	# inverse: ln(y/25.572) / 0.0966 = x, df^-1/dt = 1/(0.0966*y)
	random_offset = None
	
	equal_volume = np.array([0.821059253, 0.701839874, 0.604481024, 0.522845118, 0.454782953, 0.397055186, 0.347950518, 0.307632312, 0.271437057, 0.242110753, 0.214911277, 0.192616329, 0.171671073, 0.155542698, 0.139861868, 0.127910433, 0.115652532, 0.106316477, 0.097065429, 0.089756485, 0.082781142, 0.077048514, 0.071705703, 0.067236434, 0.063192794, 0.060011744, 0.057185781, 0.054681491, 0.052687344, 0.050997177, 0.050273861, 0.050204531, 0.051965253, 0.054669462, 0.055728143, 0.056715939, 0.052960973, 0.048262279, 0.04501948, 0.042000238, 0.041064963, 0.04049001, 0.041528099, 0.043268517, 0.047268319, 0.052921002, 0.061476518, 0.07153775, 0.083075952, 0.093187191, 0.099660587, 0.09808361, 0.090825064, 0.089826522, 0.090890027, 0.190112363, 0.59281672, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]).reshape(N, 1)
	
	V = vid(sys.argv[1])
	
	WAV = wave.open(sys.argv[3] + '.wav', 'w')
	WAV.setframerate(F)
	WAV.setnchannels(1)
	WAV.setsampwidth(2) # two bytes per sample
	audio_frames = None
	
	audio_video_frame_desync = 1/V.fps % 1/44100
	audio_video_desync = 1.0
	audio_frame_baseline = int(F/V.fps) - 1
	# can't use a pre-built array of sine values very easily, because bass (~23Hz) has a longer period than the length of a video frame, and will chop terribly
	# plus, would have to deal with more mod >_>
	
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
		
	V.initialize(float(sys.argv[2]))
	# out_video = cv2.VideoWriter(sys.argv[3] + '.mov', cv2.cv.CV_FOURCC(*'mp4v'), V.fps, (W, bottom + extension.shape[0]))
	with warnings.catch_warnings():
		warnings.filterwarnings('error', 'ooh', UserWarning)
		try:
			upper_limit = int(float(sys.argv[4]) * V.fps)
		except IndexError:
			upper_limit = float('inf')
		lower_limit = int(float(sys.argv[2]) * V.fps)
		past_heights = None
		for j in range(lower_limit, min(V.nframes, upper_limit)):
			print(j)
			v = V.read_frame()
			hsv_v = cv2.cvtColor(v, cv2.COLOR_RGB2HSV)
			hsv_frame = np.vstack((hsv_v[0:bottom, left:left+W], extension))
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
				for idx, cluster in enumerate(cluster_contours):
					A = poly_area(cluster.reshape(cluster.shape[0], cluster.shape[2]))
					if A > max_area:
						max_area = A
						max_cluster = idx
					cluster_contours[idx] = np.add(cluster, np.array([int(i * float(W)/N), 0]))
				
				extrema.append(reduce(lambda p, n: n if n < p else p, cluster_contours[max_cluster][:,:,1].reshape(cluster_contours[max_cluster].shape[0]), float('inf')))
				cv2.drawContours(hsv_frame, cluster_contours, max_cluster, (0, 255, 255), cv2.cv.CV_FILLED)
			heights = (bottom - np.array(extrema, dtype=np.float32)) / (bottom - top)
			
			tck = interpolate.splrep(np.hstack(([0.0], freqs)), np.hstack(([0.0], heights)), s=0)
			num_audio_frames = audio_frame_baseline + int(audio_video_desync + audio_video_frame_desync)
			interp_freqs = np.fft.fftfreq(num_audio_frames, d=1.0/F)
			# mags = np.zeros(num_audio_frames)
			mid_fftfreq = (num_audio_frames - 1) // 2 + 1
			top_freq = min(mid_fftfreq, int((freqs[-1] * num_audio_frames) // F + 1))
			interpd = interpolate.splev(interp_freqs[1:top_freq], tck).astype(np.complex64)
			
			np.savetxt(sys.argv[3] + '.csv', np.absolute(np.vstack((interp_freqs[1:top_freq], interpd))), delimiter=',')
			input()
			
			if random_offset == None:
				random_offset = np.random.rand(top_freq) * math.pi
			for k, i in enumerate(interpd):
				interpd[k] = cmath.rect(i, (float(j - lower_limit) / V.fps) % (1 / interp_freqs[k]) * math.pi * 2.0 + random_offset[k])
			half_freqs = np.lib.pad(interpd, (0, mid_fftfreq - top_freq), 'constant', constant_values=(0,))
			freq_domain = np.hstack(([0], half_freqs, half_freqs[::-1]))
			
			# print(mags)
			bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
			# cv2.imshow('interp', bgr_frame)
			# print(np.hstack())
			# np.savetxt('interp.csv', np.vstack((interp_freqs, np.absolute(freq_domain))), delimiter=',')
			# np.savetxt('actual.csv', np.vstack((freqs, heights)), delimiter=',')
			# input()
			
			# print(cplx)
			# np.savetxt('ifft.csv', np.real(np.fft.ifft(cplx)), delimiter=',')
			# input()
			if audio_frames == None:
				audio_frames = np.real(np.fft.ifft(freq_domain))
			else:
				audio_frames = np.hstack((audio_frames, np.real(np.fft.ifft(freq_domain))))
			
			# if past_heights != None:
			# 	num_audio_frames = audio_frame_baseline + int(audio_video_desync + audio_video_frame_desync)
			# 	ts = (np.arange(num_audio_frames) + (1.0 - audio_video_desync)) / F + float(j - lower_limit) / V.fps
			# 	cosines = np.array([np.array([math.cos(2 * math.pi * freqs[k] * ts[t]) for t in range(num_audio_frames)]) for k in range(N)])
			# 	M = (heights - past_heights) * (float(N) / V.fps)
			# 	interp = M.reshape(M.shape[0], 1).dot(ts.reshape(1, ts.shape[0])) + past_heights.reshape(past_heights.shape[0], 1)
				
			# 	if audio_frames == None:
			# 		# audio_frames = np.sum(np.multiply(np.multiply(interp, cosines), equal_volume), 0)
			# 		audio_frames = np.sum(np.multiply(interp, cosines), 0)
			# 	else:
			# 		# audio_frames = np.hstack((audio_frames, np.sum(np.multiply(np.multiply(interp, cosines), equal_volume), 0)))
			# 		audio_frames = np.hstack((audio_frames, np.sum(np.multiply(interp, cosines), 0)))
			# 	# audio_frames.append()
			# 	audio_video_desync = (audio_video_desync + audio_video_frame_desync) % 1.0
			# past_heights = heights
			# cv2.imshow(str(j), cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR))
			# bgr_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
			# cv2.imwrite(sys.argv[3] + '_frame.png', bgr_frame)
			# out_video.write(bgr_frame)
		# print((audio_frames * freq_max_amplitude).astype(np.int16))
		
		# np.savetxt('wav_out.csv', audio_frames * np.iinfo(np.int16).max, delimiter=',')
		WAV.writeframes((audio_frames * np.iinfo(np.int16).max * 50).astype(np.int16).tobytes())
		WAV.close()
		# out_video.release()
		# input()