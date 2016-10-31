import cv2
import sys
import numpy as np
import fileinput
import re
import wave
import audiotools
import math

if __name__ == '__main__':
	vid_offset = 0.021
	V_FPS = 29.97
	
	H = 500
	A = audiotools.open(sys.argv[1])
	F = A.sample_rate()
	audio_limits = (float(sys.argv[2]) * F, (float(sys.argv[4]) * F if len(sys.argv) > 4 else float(A.total_frames())))
	outfile = sys.argv[3]
	P = A.to_pcm()
	Q = P.read(int(audio_limits[0] + vid_offset * V_FPS + audio_limits[1] - audio_limits[0]))
	max_v = float('-inf')
	V = []
	
	
	for j in range(int((audio_limits[1] - audio_limits[0]) / (float(F) / V_FPS))):
		bs = [Q.frame(int(audio_limits[0] + j * (float(F) / V_FPS) + i + vid_offset * V_FPS)).to_bytes(False, True)[0:2] for i in range(int(F/V_FPS))]
		L = np.fromstring("".join(bs), dtype=np.int16)
		fft = np.absolute(np.fft.fft(L))
		max_v = max(max_v, max(fft))
		
		N = len(fft)
		midpoint = int(math.floor(N-1) // 2 + 1)
		V.append([float(v)*H for v in fft[0:midpoint]])
		# for k, v in enumerate(fft[0:midpoint]):
		# 	M[int((1.0 - float(v)/max_v)*H):,k] = 255
		# 	V.append(int((1.0 - float(v)/max_v)*H))
	
	max_w = max([len(v) for v in V])
	out_video = cv2.VideoWriter('%s.mov' % outfile, cv2.cv.CV_FOURCC(*'mp4v'), V_FPS, (max_w, H))
	
	for j, frame in enumerate(V):
		M = np.zeros((H, max_w, 3), dtype=np.uint8)
		for k, v in enumerate(frame):
			M[int(H - v/max_v):,k] = [255, 255, 255]
		# cv2.imwrite('%s.%d.png' % (outfile, j), M)
		out_video.write(M)
	out_video.release()
