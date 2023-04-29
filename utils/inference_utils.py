import sys
import os
sys.path.append(os.getcwd())

import torch
import cv2
import numpy as np

from tracker.byte_tracker import BYTETracker
from collections import OrderedDict

from utils.general import *



class VideoReader(object):
    '''
        Read .mp4 video files
    '''
    def __init__(self, file_name, grab_frame=1, batch_size=8):
        self.file_name = file_name
        self.batch_size = batch_size
        self.grab_frame = grab_frame
        self.batch_cnt = 0
        self.frame_cnt = 0

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f'Video {self.file_name} cannot be opened')
        return self

    def __next__(self):  
        batch_imgs = []

        for i in range(self.batch_size):
            # grab 1 frame
            for idx in range(self.grab_frame):
                self.cap.grab()
                self.frame_cnt += 1
            
            ret, img = self.cap.read()
            if not ret:
                raise StopIteration

            batch_imgs.append(img)
            self.frame_cnt += 1
            
        self.batch_cnt += 1
        return batch_imgs


class PoseTracker():
    '''
        Object tracking for fall detection
    '''
    def __init__(self, track_thresh=0.6, match_thresh=0.6, track_buffer=30, 
                       frame_rate=15, img_shape=(960,960)):
        
        self.bytetrack = BYTETracker(track_thresh, match_thresh, 
                                       track_buffer, frame_rate)
        self.img_shape = img_shape
        self.online_targets = None
        
        self.pose_sequences = OrderedDict()
        
    def update(self, poses):
        # input: pose estimated in 1 frame
        self.online_targets = self.bytetrack.update(poses, self.img_shape, self.img_shape)
           
        # update new objects
        for t in self.online_targets:
            # check exist id     
            obj_id = int(t.track_id)
            
            # update pose sequence for fall detection
            if obj_id in self.pose_sequences:
                self.pose_sequences[obj_id].append(t.pose.reshape(1,-1))
            else:
                # register new sequence
                self.pose_sequences[obj_id] = []
                
        # remove loss object
        for rm_obj in self.bytetrack.removed_stracks:
            rm_id = int(rm_obj.track_id)
            if rm_id in self.pose_sequences:
                del self.pose_sequences[rm_id]

                
def preprocess_batch(list_nimgs, out_img_shape=(960,960)):
    '''
        Preprocess batch images for pose estimation

        input: list cv image
        output: 
            tensor image shape [batch, 3, heigth, width] scale [0,1]
            list raw image for visualization
    '''
    out_batch = []
    raw_imgs = []
    for nimg in list_nimgs:
        nimg = letterbox(nimg, out_img_shape, stride=64)[0]
        raw_imgs.append(nimg.copy())
        
        nimg = transforms.ToTensor()(nimg)
        
        out_batch.append(nimg.unsqueeze(0))
    
    return torch.cat(out_batch, dim=0), raw_imgs                     


def detect_fall(kpt_data, model, img_shape=(960, 960)):
    '''
        input 
            kpt_data: tensor keypoints data of 1 person shape [num_frame, 38]
            model: keypoint_lstm model
            
        output: 
            predict
    '''
    # preprocess
    h,w = img_shape
    
    bbox = kpt_data[:, 0:4]
    kpt_x = kpt_data[:, 6::3]
    kpt_y = kpt_data[:, 7::3]
                     
    bbox[:, 0::2] = bbox[:, 0::2] / w 
    bbox[:, 1::2] = bbox[:, 1::2] / h  

    # center kpts data
    kpt_x = (kpt_x - 0.5*w) / (0.5*w)
    kpt_y = (kpt_y - 0.5*h) / (0.5*h)

    kpt_data = torch.cat([bbox, kpt_x, kpt_y], dim=1)
    kpt_data = kpt_data.unsqueeze(0)
    
    with torch.no_grad():
        predict = model(kpt_data)
        
    return predict 