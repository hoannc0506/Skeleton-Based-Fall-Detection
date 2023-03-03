import os
import argparse
import threading
import time
from queue  import Queue
from threading import Thread
import cv2
import numpy as np

import logging
import warnings
import subprocess as sp
from datetime import datetime
from time import sleep

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame        
        # from the stream        
        self.queue = Queue(maxsize=200)
        self.stream = cv2.VideoCapture(src)        
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)        
        (self.grabbed, self.frame) = self.stream.read()        
        self.frame = None        
        self.grabbed = False        
        # initialize the thread name        
#         command = [ 'ffmpeg',
#             # '-rtsp_transport', 'tcp',            
# #                     '-hwaccel', 'cuda',
#                     '-c:v', 'h264_cuvid',
#                     '-i', src,
#                     '-pix_fmt', 'rgb24',  # brg24 for matching OpenCV            
#                     # '-filter:v', 'fps=6',
#                     '-vsync', '0',
#                     #'-hwaccel_output_format', 'cuda',
#                     '-f', 'rawvideo',
#                     '-loglevel', 'error',
#                     'pipe:' ]
#         self.process = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8) 
        # initialize the variable used to indicate if the thread should        
        # be stopped        
        self.stopped = False        
        self.W = 1920        
        self.H = 1080        
        # self.queue.put((self.grabbed, self.frame))        
        # initialize the variable used to indicate if the thread should        
        # be stopped        
        self.stopped = False    
        
    def start(self):
        # start the thread to read frames from the video stream        
        Thread(target=self.update, args=(), daemon=True).start()
        return self    
    
    def update(self):
        # keep looping infinitely until the thread is stopped        
        while True:
            # if the thread indicator variable is set, stop the thread            
            if self.stopped:
                return            
            # otherwise, read the next frame from the stream            
            (self.grabbed, self.frame) = self.stream.read()            
#             buffer = self.process.stdout.read(self.W*self.H*3)
#             if len(buffer) != self.W*self.H*3:
#                 continue            
#             img = np.frombuffer(buffer, np.uint8).reshape(self.H, self.W, 3)
#             self.grabbed, self.frame = True, img            
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            # if self.queue.qsize() == 200:            
            #     self.queue.get(0)           
            
            
            self.queue.put((self.grabbed, self.frame))
            
    def read(self):
        # return the frame most recently read        
        # return self.grabbed, self.frame        
        if self.queue.qsize()==0:
            time.sleep(2)
        try:
            grab, frame = self.queue.get(0)
        except:
            return (False, None), 0        
        return (grab, frame), self.queue.qsize()
    
    def stop(self):
        # indicate that the thread should be stopped        
        self.stopped = True

if __name__== "__main__": 
    stream = WebcamVideoStream("rtmp://live-10-hcm.fcam.vn:1956/63eef9bd282e0ab5mye8?t=1671440944&tk=36e7618edfb8c695c9ea5b5245778dce8a57bc45bb93e19098790fd03eeab6aa/KeYm2RCH-AhrStRPD-9g5Kioe1-PbAvhlmw-v2")
    
    out_video = []
    frame_cnt = 0
  
    try:
        while True:
            data, _ = stream.read()

            ret, frame = data
            print('here')
            if not ret:
                time.sleep(0.1)
                continue  
                
            print('streaming')
            out_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_cnt +=1

    except KeyboardInterrupt:   
        import moviepy.editor as mpy 
        vid = mpy.ImageSequenceClip(out_video, fps=20)
        vid.write_videofile('demo/demo_raw_cam.mp4')
        


        
