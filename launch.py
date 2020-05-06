from pipeline import pipeline
from segment import segment
from meanshift import meanshift
from lkcomponents import lkcomponents
from lktemplate import lktemplate
from camshift import camshift
from meanshiftdefault import meanshiftdefault
from KCFtrack import KCFtrack
from lkflow import lkflow
from lkcomponentshsv import lkcomponentshsv
from meanshiftsaturation import meanshiftsaturation
from meanshiftspacekernel import meanshiftspacekernel

import cv2
import pandas as pd 
import numpy as np
from loadvideo import loadvideo

import json



def launch():
    config = readconfig()
    inputfile = config["input"]
    outputfile = config["output"]
    video = loadvideo(inputfile)



    algorithm = menu(video,config)


    points = pipeline(algorithm,outputfile)
    launch()

def menu(video,config):
    options = ['lkcomponents','lktemplate','kernelmeanshift','camshift',
               'playback','readconfig','dispconfig','meanshift',
               'kcf','quit','lkflow','lkcomponentshsv', '']
    while(1):
        name = input("pick algorithm to use for image tracker: ")
        if(name == 'help'):
            print(options)    
        elif(name == options[0]):
            return lkcomponents(video,config)
        elif(name == options[1]):
            return lktemplate(video,config)
        elif(name == options[2]):
            #pass
            # broken: tried to implement Comaniciu algorithm with kernel based histogram comparison
            return meanshiftspacekernel(video,config)
        elif(name == options[3]):
            return camshift(video,config)
        elif(name == options[4]):
            outputfile = config["output"]
            playback(outputfile)
        elif(name == options[5]):
            launch()
        elif(name == options[6]):
            print(config)
        elif(name == options[7]):
            return meanshiftdefault(video,config)
        elif(name == options[8]):
                print('not avaliable')
                # not done, time issues
                #return KCFtrack(video,config)
        elif(name == options[9]):
            exit()
        elif(name == options[10]):
            return lkflow(video,config)
        elif(name == options[11]):
            return lkcomponentshsv(video,config)
        else:
            print("invalid selection type (help) for help")

def readconfig():
    f = open("params.json", "r")
    string = f.read() 
    config = json.loads(string)
    return config

def playback(outputfile):
    video = loadvideo(outputfile)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) 
    for i in range(n_frames-2):
        ret, frame = video.read()
        if(ret):
            cv2.imshow('playback',frame)
        k = cv2.waitKey(30) & 0xff
        if(k==27):
            break
    video.release()
    cv2.destroyAllWindows()




if __name__=='__main__':
    launch()


# goal find where failure points are

# log
# design issues of parameter config
# sometimes the grayscale template follows the car and then stops following
# static features overwhelming component based tracking
#    - one solution is to use the manual roi to find features on image crop.
# requirements for tracking multiple images.
# the biggest challange is the issue of what to do if the image goes outside the frame. 
#    - change to another template
#    -

# 4/16/2020
# finally found a better metric for points to be removed
# need to fix issue with video not continueing when there aren't any points


# algorithm process
#  
#
#

# sources
# script design: thanks to design patterns
# input source: https://www.geeksforgeeks.org/taking-input-in-python/
# lkflow example: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# another lkflow example: https://nanonets.com/blog/optical-flow/ 
# lkflow forward backward error: https://jayrambhia.com/blog/lucas-kanade-tracker 
# tracks: https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py 
# fb error: https://www.sciencedirect.com/topics/computer-science/backward-error 
#  
# meanshift process: https://docs.opencv.org/3.4/d7/d00/tutorial_meanshift.html
# Kalman Filter process: https://docs.opencv.org/trunk/de/d70/samples_2cpp_2kalman_8cpp-example.html#_a7 


# examples of video working shows a sanity check of some failure points with occlusion which I saw in the traffic example
# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

# file write: 
# https://www.guru99.com/reading-and-writing-files-in-python.html 
# https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file


# manual selecting roi for template
# https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/

#lkflow function docs
# https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html?highlight=calcopticalflowpyrlk#cv2.calcOpticalFlowPyrLK



#  snakes library
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html



#https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/



#https://stackoverflow.com/questions/16343752/numpy-where-function-multiple-conditions




#Fast and Robust Object Tracking Using Tracking Failure Detection in Kernelized Correlation Filter

# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
# http://vision.ucsd.edu/~bbabenko/data/miltrack_cvpr09.pdf 


# https://docs.opencv.org/3.4/de/df4/tutorial_js_bg_subtraction.html
#




# mask issue reason: https://stackoverflow.com/questions/21782420/difference-between-hsv-and-hsv-full-in-opencv


# getting some understanding of the bhattacharyya distance
# https://gist.github.com/jstadler/c47861f3d86c40b82d4c
# https://www.encyclopediaofmath.org/index.php/Bhattacharyya_distance



# Bhattacharyya comparision for histograms
# https://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/


# ancillery source for distance metrics
# http://users.umiacs.umd.edu/~ramani/pubs/YDD_CVPR_05.pdf



# https://stackoverflow.com/questions/11209115/creating-gaussian-filter-of-required-length-in-python


#kernel issues
# https://github.com/scikit-learn/scikit-learn/issues/442
# http://www.ipol.im/pub/art/2019/255/article_lr.pdf 



#http://www.cse.psu.edu/~rtc12/CSE598G/introMeanShift.pdf


# histogram change is based on Indexing via color histograms 
# the paper is what is used by opencv
# Swain, M.J. (1990). "Indexing via color histograms". 
# [1990] Proceedings Third International Conference on Computer Vision , 
# p. 390.

#Bradski, G.R., 
# "Real time face and object tracking as a 
# component of a perceptual user interface,"


#Y. Wu, J. Lim and M. Yang, "Object Tracking Benchmark," 
# in IEEE Transactions on Pattern Analysis and Machine Intelligence, 
# vol. 37, no. 9, pp. 1834-1848, 1 Sept. 2015.





# video write (playback feature)
# https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python
# http://arahna.de/opencv-save-video/ 
# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
# 