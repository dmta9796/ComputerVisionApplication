import numpy as np
import cv2
import os
import math

class lkcomponents():
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
        self.feature_params["maxCorners"] = config['lkcomponents']['feature_params_maxCorners']
        self.feature_params["qualityLevel"] = config['lkcomponents']['feature_params_qualityLevel']
        self.feature_params["minDistance"] = config['lkcomponents']['feature_params_minDistance']
        self.feature_params["blockSize"] = config['lkcomponents']['feature_params_blockSize']
        self.lk_params["winSize"] = tuple(config['lkcomponents']["lk_params_winSize"])
        self.lk_params["maxLevel"] = config['lkcomponents']["lk_params_maxLevel"]
        self.lk_params["criteria"] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,config['lkflow']["lk_params_criteria_EPS"],config['lkflow']["lk_params_criteria_COUNT"])

        self.lossthreshold = config['lkcomponents']['lossthreshold']        

    def start(self):
        self.tracks = []
        self.track_len = 10
        self.detect = 5
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))


        ret,first_frame = self.video.read()
        r = cv2.selectROI(first_frame)
        (x,y,dx,dy) = r 
        self.box = r

        # many templates
        self.p0list = []
        self.template_gray = cv2.cvtColor(first_frame,cv2.COLOR_RGB2GRAY)
        region = self.template_gray[y:y+dy,x:x+dx]
        points = cv2.goodFeaturesToTrack(region, **self.feature_params)
        points = np.float32(points + [x,y])
        
        self.p0list = points
        self.p0size = len(self.p0list)
        self.p0flags = np.ones(len(self.p0list))
        self.p0lastgoodI = np.zeros(len(self.p0list))


    def step(self,step):
        ret, frame = self.video.read()
        if(ret):
            gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            vis = frame.copy()
            if(len(self.p0list) != 0):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st1, err = cv2.calcOpticalFlowPyrLK(self.template_gray, gray, self.p0list, None, **self.lk_params)
                p0r,st2,error = cv2.calcOpticalFlowPyrLK(gray,self.template_gray,p1,None,**self.lk_params)
                d = abs(self.p0list-p0r).reshape(-1,2).max(-1)
                #print(d)
                # Select good points
                bound = st1 == 1
                good = d < 1

                
                #prev_average = np.sum(good_prev_corners,axis=0)
                #new_average = np.sum(good_new_corners,axis=0)
                #diff_avg = new_average-prev_average

                # compute average of good points
                idx = np.where(good)
                good_prev_points = self.p0list[idx]
                good_new_points = p1[idx]
                prev_average = np.sum(good_prev_points,axis=0)
                new_average = np.sum(good_new_points,axis=0)
                diff_avg = new_average-prev_average
                
                data = []
                for i,(new,good,flag) in enumerate(zip(p1,good,self.p0flags)):
                    x,y = new.ravel()
                    x = int(x)
                    y = int(y)
                    if(good and flag):
                        # update p0list
                        self.p0list[i] = p1[i]

                        # display good points
                        self.mask = cv2.circle(frame,(x,y),5,(255,0,0))
                        dx = (int)(self.lk_params["winSize"][0]/2)
                        dy = (int)(self.lk_params["winSize"][1]/2)
                        rect = cv2.rectangle(frame,(int(x)-dx,int(y)-dy),(int(x)+dx,int(y)+dy),(255,0,0))
                        data.append((x,y))
                        if(x >= 0 and x< self.width and y >= 0 and y < self.height ):
                            self.p0lastgoodI[i] = gray[y,x]
                    else: 
                        # update with the unabscured points
                        self.p0list[i] = self.p0list[i] + diff_avg
                        self.p0flags[i] = 0
                        
                        if(x >= 0 and x< self.width and y >= 0 and y < self.height):
                            I = gray[y,x] 
                            if(self.p0lastgoodI[i]==I):
                                self.p0flags[i]=1
                        
                        if(np.sum(self.p0flags)/len(self.p0flags)<self.lossthreshold):
                            # declare object lost
                            self.p0list = []
                            break

                    # Now update the previous frame and previous points
                    self.template_gray = gray
                    #self.p0list = p1.reshape(-1,1,2)

            else: 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (10, 10) 
                fontScale = 0.25
                color = (0, 0, 255) 
                thickness = 1
                frame = cv2.putText(frame, 'Object Lost', org, font,  
                fontScale, color, thickness, cv2.LINE_AA) 
            
            cv2.imshow('frame',frame)
            return frame


