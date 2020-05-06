import numpy
import cv2
def loadvideo(filename):
    cap = cv2.VideoCapture(filename)
    return cap