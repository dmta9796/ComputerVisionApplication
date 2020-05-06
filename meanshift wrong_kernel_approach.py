import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi, sqrt, exp

class meanshift():
    video = None
    n_frames = 0
    roi_hist = None
    term_crit = None

    def __init__(self,video,config):
        self.video = video 
        self.binstep = config["meanshift"]["binstep"]
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) 
        self.w0 = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h0 = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi_hist = None
        self.term_crit = None
        self.transform = cv2.COLOR_BGR2RGB
        self.maskrange = tuple(config["meanshift"]["maskrange"])
    
    def start(self):
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange

        ret,frame = self.video.read()

        r = cv2.selectROI(frame)

        x = r[0] #175 #(int)(np.random.uniform(0,w0-w))
        y = r[1] #138 #(int)(np.random.uniform(0,h0-h))
        w = r[2]
        h = r[3]

        self.window=(x,y,w,h)
        # set up the ROI for tracking
        roi = frame[y:y+h, x:x+w]
        hsv_roi =  cv2.cvtColor(roi, self.transform)
        self.mask = cv2.inRange(hsv_roi, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],self.mask,[upperhue],[lowerhue,upperhue])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.prev_frame = frame


    def step(self,step):
        ret,frame = self.video.read()
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        x,y,w,h = self.window
        if(ret):
            # image 
            hsv = cv2.cvtColor(frame, self.transform)
            # old and new roi
            new_roi = frame[y:y+h,x:x+w]
            old_roi = self.prev_frame[y:y+h,x:x+w]
            # new and old mask
            mask_new = cv2.inRange(new_roi, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            mask_old = cv2.inRange(old_roi, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            # new and old histogram
            hist_old = cv2.calcHist([old_roi],[0],mask_old,[180],[0,180])
            hist_new = cv2.calcHist([new_roi],[0],mask_new,[180],[0,180])
            # normalize
            cv2.normalize(hist_new,hist_new,0,255,cv2.NORM_MINMAX)
            cv2.normalize(hist_old,hist_old,0,255,cv2.NORM_MINMAX)

            # hsv = cv2.cvtColor(frame, self.transform)
            # newmask = cv2.inRange(hsv, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            # hsv_old = cv2.cvtColor(self.prev_frame,self.transform) 

            # hist_old = cv2.calcHist([hsv_old],[0],self.mask,[180],[0,180])
            # cv2.normalize(hist_old,hist_old,0,255,cv2.NORM_MINMAX)

            # hist_new = cv2.calcHist([hsv],[0],self.mask,[180],[0,180])
            # cv2.normalize(hist_new,hist_new,0,255,cv2.NORM_MINMAX)


            distribution = self.dist(self.roi_hist,hist_new)
            dst = cv2.calcBackProject([hsv],[0],distribution,[0,180],self.binstep)
            dst_compare = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],self.binstep)
            # apply meanshift to get the new location
            ret, self.window = cv2.meanShift(dst, self.window, self.term_crit)
            # Draw it on image
            x,y,w,h = self.window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            # cv2.imshow('hist',self.roi_hist)
            cv2.imshow('dst',dst)
            #cv2.imshow('dst_compare',dst_compare)
            cv2.imshow('img2',img2)
            self.prev_frame = frame
            return (x,y)

    def dist(self,hist1,hist2):
        kernel = self.gauss(180,10)
        adist = cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
        diff = abs(hist1-hist2)
        result = hist1 #np.convolve(kernel,diff.flatten())*adist
        return result

    def box(self,n=11):
        return np.ones(n)

    def gauss(self,n=11,sigma=1):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]