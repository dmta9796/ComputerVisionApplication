import numpy as np
import cv2
import math

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
        self.transform = cv2.COLOR_BGR2HSV
        self.maskrange = tuple(config["meanshift"]["maskrange"])
        self.d_tol = 10
        self.h = 1
    
    def start(self):
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange

        ret,frame = self.video.read()

        r = cv2.selectROI(frame)

        x = r[0] #175 #(int)(np.random.uniform(0,w0-w))
        y = r[1] #138 #(int)(np.random.uniform(0,h0-h))
        w = r[2] if (r[2])%2 == 1 else r[2]-1 # must be odd
        h = r[3] if (r[3])%2 == 1 else r[3]-1 # must be odd

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
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        (x,y,w,h)= self.window
        ret,frame = self.video.read()
        if(ret):
            #roi = frame[y:y+h, x:x+w]
            hsv = cv2.cvtColor(frame,self.transform)
            #mask = cv2.inRange(hsv, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            #hist = cv2.calcHist([hsv],[0],mask,[upperhue],[lowerhue,upperhue])
            #d = cv2.compareHist(hist,self.roi_hist,cv2.HISTCMP_BHATTACHARYYA)

            ret, self.window = self.meanshiftpaper(hsv,self.window,self.term_crit)



        #     hsv = cv2.cvtColor(frame, self.transform)
        #     dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],self.binstep)

        #     cv2.imshow('space',dst)
        #     # apply meanshift to get the new location
        #     ret, self.window = self.meanshiftalgorithm(dst,self.window,self.term_crit) #cv2.meanShift(dst, self.window, self.term_crit)
            # Draw it on image
            x,y,w,h = self.window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            #cv2.imshow('hist',self.roi_hist)
            cv2.imshow('img2',img2)
            return img2


    # histogram change is based on Indexing via color histograms 
    # the paper is what is used by opencv

    # from Feature Extraction & Image Processing for Computer Vision.
    #computer gradient with the difference
    # find maxima
    # update mean
    # backproject to image
    def meanshiftpaper(self,image,window,term_crit):
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        _,iterations,epsilon = term_crit
        x,y,dx,dy = window

        qs = self.roi_hist 

        center = (int(dx/2),int(dy/2))
        xc,yc = center
        qssize = len(qs)
        qscenter = 0
        K = self.gauss_kernel(qssize,qscenter,self.h)
        qs = K*qs


        # density is histogram
        region = image[y:y+dy,x:x+dx]
        sat = region[:,:,1]
        mask = cv2.inRange(region, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
        hist = cv2.calcHist([sat],[0],mask,[upperhue],[lowerhue,upperhue])
        histcenter = np.average(sat)
        histsize = len(hist)
        K = self.gauss_kernel(histsize,histcenter,self.h)
        #print(K)
        q = K*hist

        for (a,b) in zip(q,qs):
            print(a,b)
        w = [np.sqrt(q_i/qs_i) if qs_i!=0 else 0 for (q_i,qs_i) in zip(q[0],qs[0])]
        w = np.nan_to_num(w)

        x_i = self.getcoords(region) + [x,y]





    def meanshift(self,image,window,term_crit):
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        _,iterations,epsilon = term_crit
        x,y,dx,dy = window

        q = self.roi_hist
        center = (int(x+dx/2),int(y+dy/2))
        ret= False
        for i in range(iterations):
            region = image[y:y+dy, x:x+dx]
            #mask = cv2.inRange(region, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            #hist = cv2.calcHist([region],[0],mask,[upperhue],[lowerhue,upperhue])
            w = cv2.calcBackProject([region],[0],q,[0,180],self.binstep)
            x_i = self.getcoords(image)[y:y+dy, x:x+dx]
            g_i = self.gradient(x_i,center,self.h)

            numarr = np.array([w*x_i[:,:,0]*g_i[:,:,0],w*x_i[:,:,1]*g_i[:,:,1]])
            demarr = np.array([w*g_i[:,:,0],w*g_i[:,:,1]])
            num = np.array([np.sum(numarr[0]),np.sum(numarr[1])])
            dem = np.array([np.sum(demarr[0]),np.sum(demarr[1])])
            delta_x = num/dem
            delta_x = np.nan_to_num(delta_x)

            old_x = center
            center = center + delta_x
            if(self.dist2(old_x,center)<epsilon):
                ret = True
                break

        return (ret,window)
    

    def meanshiftalgorithm(self,image,window,term_crit):
        # config params
        lowerhue,lowersat,lowervalue,upperhue,uppersat,uppervalue = self.maskrange
        _,iterations,epsilon = term_crit
        x,y,dx,dy = window


        y_hat_0 = (int(dx/2),int(dy/2))
        d_tol = 5
        h = 1
        q_hat = self.roi_hist # histogram

        C = np.sum(q_hat)
        q_hat =(1/C)*q_hat




        for i in range(iterations):
            #step 1: init
            # get center 
            y_hat_0 = (int(dx/2),int(dy/2))
            #compute p_u(y_hat_0)
            region = image[y:y+dy, x:x+dx]
            x_i = self.getcoords(region,y_hat_0)
            mask = cv2.inRange(region, np.array((lowerhue, lowersat,lowervalue)), np.array((upperhue,uppersat,uppervalue)))
            p_hat_0 = cv2.calcHist([region],[0],mask,[upperhue],[lowerhue,upperhue])
            p_hat_0 = p_hat_0/np.sum(p_hat_0)

            # get q_hat_u
            # find rho(p_hat(y_hat_0))
            rho0 = self.bhattacharyya_coeff(p_hat_0,q_hat)

            # compute kernel 
            norm = self.dist(y_hat_0,x_i,self.h)
            K = self.kernel(norm)
            q_hat = np.convolve(q_hat.flatten(),K)
            p_hat_0 = np.convolve(p_hat_0.flatten(),K)
            
            #step 2: find weights w_i 
            # idx = 1
            # delta = 1
            #u = x_i[y_hat_0[1]][y_hat_0[0]][2]
            b = cv2.calcBackProject([region],[0],np.floor(q_hat*C),[0,180],self.binstep)  
            u = region[y_hat_0[1]][y_hat_0[0]][2]
            diff = (b-u)
            #cv2.imshow('blah',b)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            delta = np.where(b==u,1,0)
            w_i = []
            for i in range(180):
                w_i.append(np.sqrt(q_hat[i]/p_hat_0[i])*diff)
            w_i = np.sum(w_i,axis=0)

            #step 3: meanshift
            numarr = region[:,:,1]*w_i*K
            demarr = w_i*K
            num = np.sum(numarr)
            dem = np.sum(demarr)
            y_hat_1 = num/dem 

            #step 4: update p_hat_y and get new comparison rho1
            p_hat_1 = cv2.calcHist([region],[0],mask,[upperhue],[lowerhue,upperhue])

            rho1 = cv2.compareHist(p_hat_1,self.roi_hist,cv2.HISTCMP_BHATTACHARYYA)
            # step 5 not needed except in bad cases
            # while rho1 < rho0:
            #     y_hat_1 = 0.5*(y_hat_1+y_hat_0)

            if(self.dist2(y_hat_0,y_hat_1)<epsilon):
                break
            else:
                y_hat_0 = y_hat_1
        
        x,y = y_hat_0
        window = (int(x-dx/2),int(y-dy/2),dx,dy)
        return window

    def gethistograms(self):
        pass  


    def getcoords(self,image):
        a,b,_ = image.shape
        coords = np.zeros((a,b,2))
        for i,row in enumerate(coords):
            for j,_ in enumerate(row):
                coords[i,j] = [i,j]
        return coords



    def neighborlist(self,region,y_hat_0,d_tol=5):
        neighbors = []
        point = y_hat_0
        for x,row in enumerate(region):
            for y,data in enumerate(row):
                point2 = [x,y]
                d = self.dist(point,point2,self.h)
                if(d<d_tol):
                    neighbors.append(point2)
        return neighbors


    def dist(self,p1,p2,h):
        acc = 0 
        for (x_i,x_j) in zip(p1,p2):
            acc = acc + (x_i-x_j)**2
        result = np.sqrt(acc/h)
        return result

    def dist2(self,p1,p2):
        acc = 0 
        for (x_i,x_j) in zip(p1,p2):
            acc = acc + (x_i-x_j)**2
        result = np.sqrt(acc)
        return result

    def bhattacharyya_coeff(self,hist1,hist2):
        score = cv2.compareHist(hist1,hist2,cv2.HISTCMP_BHATTACHARYYA)
        return np.sqrt(1-score)


    def gauss_kernel(self,size,center,sigma=1):
        kernel = np.zeros(size)
        kernel = [(1/(math.pi*sigma*sigma))*math.exp(-(i-center)**2) for i in range(size)]
        return kernel

    def gradient(self,x_i,center,h):
        g = np.zeros(x_i.shape)
        for i,row in enumerate(x_i):
            for j,point in enumerate(row):
                norm = self.dist(point,center,h)
                val = self.kernel(norm)
                g[i,j] = val

        return g




#http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/
# https://github.com/zbxzc35/Meanshift-1


# equation 25: weighting histogram better explaination
# https://pdfs.semanticscholar.org/777d/a067d56dd2e83cc15cf0217c4bba6d9ebe3c.pdf

# reshape
# https://stackoverflow.com/questions/18757742/how-to-flatten-only-some-dimensions-of-a-numpy-array



# survey of methods
# http://www.cs.ucf.edu/courses/cap6412/fall2009/papers/Object_Tracking.pdf



# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4279509/
# http://www.cse.psu.edu/~rtc12/CSE598G/introMeanShift.pdf 



# https://docs.opencv.org/master/dc/df6/tutorial_py_histogram_backprojection.html

# https://stackoverflow.com/questions/47369579/how-to-get-the-gaussian-filter