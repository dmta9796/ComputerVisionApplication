import numpy as np
import cv2
import math

class meanshiftspacekernel():
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
        self.transform = cv2.COLOR_BGR2HSV
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
        mask = cv2.inRange(hsv_roi, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
        self.roi_hist = cv2.calcHist([hsv_roi],[0],mask,[upperhue],[lowerhue,upperhue])
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.prev_frame = frame


    def step(self,step):
        ret,frame = self.video.read()
        if(ret):
            x,y,w,h = self.window
            center = (int(x+w/2),int(y+h/2))
            hsv = cv2.cvtColor(frame, self.transform)
            dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],self.binstep)

            K = self.kernel(center,dst.shape)
            dst= dst*K
            # apply meanshift to get the new location
            ret, self.window = cv2.meanShift(dst, self.window, self.term_crit)
            # Draw it on image
            x,y,w,h = self.window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
           # cv2.imshow('hist',self.roi_hist)
            cv2.imshow('space',dst)
            cv2.imshow('img2',img2)
            return img2

    def kernel(self,center,size):
        result = self.box(100,center,size)
        return result

    def box(self,n,center,size):
        result = np.zeros(size).astype("uint8")
        for i,row in enumerate(result):
            for j,_ in enumerate(row):
                point = (j,i)
                distance = self.dist(center,point)
                if(distance < n):
                    result[i,j] = int(1)

        return result
    
    def dist(self,p1,p2):
        acc = 0 
        for (x_i,x_j) in zip(p1,p2):
            acc = acc + (x_i-x_j)**2
        result = np.sqrt(acc)
        return result

    def gauss(self,n=11,sigma=1):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]


#http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/