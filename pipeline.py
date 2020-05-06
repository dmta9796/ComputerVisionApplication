import cv2

def pipeline(algorithm,outputfile):
    width = int(algorithm.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(algorithm.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (width,height))
    
    
    points = []
    algorithm.start()
    frames = algorithm.n_frames
    
    for i in range(frames-2):
        frame = algorithm.step(i)
        if(frame is not None):
            output.write(frame)
        k = cv2.waitKey(30) & 0xff
        if(k==27):
            break
    

    algorithm.video.release()
    output.release()
    cv2.destroyAllWindows()
    return output


