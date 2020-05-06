import numpy as np
import cv2

class camshift():
    video = None
    n_frames = 0
    roi_hist = None
    term_crit = None

    def __init__(self,video,config):
        self.binstep = config["camshift"]["binstep"]
        self.maskrange = tuple(config["camshift"]["maskrange"])


        self.video = video 
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) 
        self.w0 = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h0 = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.roi_hist = None
        self.term_crit = None
        self.transform = cv2.COLOR_BGR2HSV
    
    def start(self):
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        ret,frame = self.video.read()

        r = cv2.selectROI(frame)

        x = r[0] #175 #(int)(np.random.uniform(0,w0-w))
        y = r[1] #138 #(int)(np.random.uniform(0,h0-h))
        w = r[2]
        h = r[3]

        self.window=(x,y,w,h)
        print(self.window)

        # set up the ROI for tracking
        roi = frame[y:y+h, x:x+w]
        hsv_roi =  cv2.cvtColor(roi, self.transform)
        mask = cv2.inRange(hsv_roi, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[upperhue],[lowerhue,upperhue])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, 1 )


    def step(self,step):
        ret,frame = self.video.read()
        if(ret):
            hsv = cv2.cvtColor(frame, self.transform)
            dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],self.binstep)
            # apply meanshift to get the new location
            ret, self.window = cv2.CamShift(dst, self.window, self.term_crit)
            # Draw it on image
            x,y,w,h = self.window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
           # cv2.imshow('hist',self.roi_hist)
            cv2.imshow('space',dst)
            cv2.imshow('img2',img2)

            return img2