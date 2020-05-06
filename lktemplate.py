import numpy as np
import cv2
import os

class lktemplate():
    points = dict()
    video = None
    n_frames = 0

    lk_params = dict( winSize  = (25,25),
                  status = 1,
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    def __init__(self,video,config):
        self.video = video
        self.lk_params["maxLevel"] = config['lktemplate']["lk_params_maxLevel"]
        self.lk_params["criteria"] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,config['lktemplate']["lk_params_criteria_EPS"],config['lktemplate']["lk_params_criteria_COUNT"])
    
    
    def start(self,maxCorners=200):
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) 


        self.transforms = np.zeros((self.n_frames-1, 3), np.float32)
        ret,first_frame = self.video.read()
        self.template_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

        self.w = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #self.p0 = cv2.goodFeaturesToTrack(self.template_gray, **self.feature_params)
        self.color = np.random.randint(0,255,(maxCorners,3))

        # select image
        r = cv2.selectROI(first_frame)
        print(r)
        self.p0 = np.array([[[np.float32(r[0]+r[2]/2),np.float32(r[1]+r[3]/2)]]])
        #print(self.p0)
        self.lk_params["winSize"] = (r[2],r[3])
        #self.lk_params["maxLevel"]= 2
        #self.lk_params["criteria"] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)

        self.mask = np.zeros_like(self.template_gray)



    def step(self,step):
        ret, frame = self.video.read()
        data =None
        if(ret):
            if(self.p0.size != 0):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.template_gray, gray, self.p0, None, **self.lk_params)
                p0r,st,error = cv2.calcOpticalFlowPyrLK(gray,self.template_gray,p1,None,**self.lk_params)
                d = abs(self.p0-p0r).reshape(-1,2).max(-1)
                #print(d)
                # Select good points
                idx = np.where(st==1)[0]
                good_new_corners = p1[idx]
                good_prev_corners = self.p0[idx]

                data = []
                for i,new in enumerate(good_new_corners):
                    a,b = new.ravel()
                    self.mask = cv2.circle(frame,(a,b),5,self.color[i].tolist())
                    da = (int)(self.lk_params["winSize"][0]/2)
                    db = (int)(self.lk_params["winSize"][1]/2)
                    rect = cv2.rectangle(frame,(int(a)-da,int(b)-db),(int(a)+da,int(b)+db),self.color[i].tolist())
                    data.append((a,b))
                


                
                # Now update the previous frame and previous points
                self.template_gray = gray
                self.p0 = good_new_corners.reshape(-1,1,2)

            else: 
                font = cv2.FONT_HERSHEY_SIMPLEX 
                org = (10, 10) 
                fontScale = 0.25
                color = (0, 0, 255) 
                thickness = 1
                frame = cv2.putText(frame, 'Template Lost', org, font,  
                fontScale, color, thickness, cv2.LINE_AA) 
            cv2.imshow('frame',frame)
            return frame



    # pipeline was my idea
    # the tips for lkflow is from https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html