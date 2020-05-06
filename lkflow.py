import numpy as np
import cv2
import os
import math

class lkflow():
    points = dict()
    video = None
    n_frames = 0
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 10,
                       blockSize = 3 )

    lk_params = dict( winSize  = (31,31),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self,video,config):
        self.video = video
        self.feature_params["maxCorners"] = config['lkflow']['feature_params_maxCorners']
        self.feature_params["qualityLevel"] = config['lkflow']['feature_params_qualityLevel']
        self.feature_params["minDistance"] = config['lkflow']['feature_params_minDistance']
        self.feature_params["blockSize"] = config['lkflow']['feature_params_blockSize']
        self.lk_params["winSize"] = tuple(config['lkflow']["lk_params_winSize"])
        self.lk_params["maxLevel"] = config['lkflow']["lk_params_maxLevel"]
        self.lk_params["criteria"] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,config['lkflow']["lk_params_criteria_EPS"],config['lkflow']["lk_params_criteria_COUNT"])
    
    def start(self):
        self.tracks = []
        self.track_len = 10
        self.detect = 5
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) 


        ret,first_frame = self.video.read()
        r = cv2.selectROI(first_frame)
        (x,y,dx,dy) = r 
        self.box = r


    def step(self,step):
        ret, frame = self.video.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        vis = frame.copy()
        if(len(self.tracks) > 0):
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.template_gray, gray,p0, None, **self.lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(gray, self.template_gray, p1, None, **self.lk_params)

            d = abs(p0-p0r).reshape(-1,2).max(-1)
            new_tracks = []
            good = d < 1
            for tr,(x,y),good_flag in zip(self.tracks,p1.reshape(-1,2),good):
                if not good_flag:
                    continue
                
                tr.append((x,y))
                if len(tr)>self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis,(x,y),2,(0,255,0),-1)

            self.tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
            self.template_gray = gray
            self.moveregion()

        if(step % self.detect == 0):
            self.getNewPoints(gray)
        self.template_gray = gray
        (x,y,dx,dy)= self.box
        cv2.rectangle(vis,(int(x),int(y)),(int(x+dx),int(y+dy)),(255,0,0))
        cv2.imshow('frame',vis)
        return frame


    def getNewPoints(self,gray):
        r = self.box
        (px,py,dx,dy) = r
        #region = gray[int(px):int(px+dx), int(py):int(py+dy)]
        
        #cv2.rectangle(gray,(int(px),int(py)),(int(px+dx),int(py+dy)),(255,0,0))
        #cv2.imshow('test',gray)
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
        p0 = cv2.goodFeaturesToTrack(gray, mask = mask, **self.feature_params)
        
        if(p0 is not None and p0.size != 0):
            for x, y in np.float32(p0).reshape(-1, 2):
                self.tracks.append([(x,y)])

    def moveregion(self):
        (current,past) = self.getrecent()
        (x,y,dx,dy) = self.box
        diff = np.array(current) - np.array(past)
        if(len(diff)!=0):
            avg = np.average(diff,axis=0)
            (x,y,dx,dy) = self.box
            x = x + avg[0]
            y = y + avg[1]
            self.box = (x,y,dx,dy)
    
    def getrecent(self):
        tracks = np.array(self.tracks)
        current = []
        past = []
        (x,y,dx,dy) = self.box
        for track in tracks:
            c = track[-1]
            p = track[-2]
            if(c[0]>=x and c[0]<=(x+dx) and c[1]>=y and c[1]<=(y+dy)):
                current.append(c)
            if(p[0]>=x and p[0]<=(x+dx) and p[1]>=y and p[1]<=(y+dy)):
                past.append(p)
        length = np.min([len(current),len(past)])
        past = past[0:length]
        current = current[0:length]
        return (current,past)
            
    def findpoints(self,c,p):
        (x,y,dx,dy) = self.box
        currentidx = np.where((c[:,0]>=x) & (c[:,0]<=(x+dx)) & (c[:,1]>=y) & (c[:,1]<=(y+dy)))[0]
        pastidx = np.where((p[:,0]>=x) & (p[:,0]<=(x+dx)) & (p[:,1]>=y) & (p[:,1]<=(y+dy)))[0]
        current = c[currentidx]
        past = p[pastidx]
        return (current,past)

