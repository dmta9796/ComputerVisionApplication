import numpy as np
import cv2
import os

class snakes():
    points = dict()
    video = None
    n_frames = 0
    def __init__(self,video):
        self.video = video
    
    def start(self,maxCorners=1,qualityLevel= 0.01,minDistance = 10, blockSize = 3, winSize = (31,31), maxLevel= 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        pass



    def step(self,step):
        ret, frame = self.video.read()
        if(ret):
            pass



    # pipeline was my idea
    # the tips for lkflow is from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html