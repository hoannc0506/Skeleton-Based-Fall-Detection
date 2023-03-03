# import trt befor torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import cv2
import os
import time
import numpy as np
import argparse
import logging

from lstm_models import KeypointsLSTM
from pose_utils import *

from tracker.byte_tracker import BYTETracker
from collections import OrderedDict

from tensorrt_inference import TRTInferenceEngine
from stream_webcam import WebcamVideoStream
 
def preprocess_batch(list_nimgs, out_img_shape=(768,960)):
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
      
    
class PoseTracker():
    '''
        Object tracking for fall detection
    '''
    def __init__(self, track_thresh=0.5, match_thresh=0.7, track_buffer=30, 
                       frame_rate=30, img_shape=(768,960)):
        
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
                
                     
                
def fall_detection(kpt_data, model, img_shape=(768, 960)):
    '''
        input: list tensor keypoints data of 1 person, model 
        output: is falling detected
    '''
    # preprocess
    bbox = xyxy2xywh(kpt_data[:, :4])
    kpt_x = kpt_data[:, 6::3]
    kpt_y = kpt_data[:, 7::3]
    
    # center kpts data
    h,w = img_shape
    bbox = bbox / torch.tensor([w,h,w,h], device=kpt_data.device)
    kpt_x = (kpt_x - 0.5*w) / (0.5*w)
    kpt_y = (kpt_y - 0.5*h) / (0.5*h)

    kpt_data = torch.cat([bbox, kpt_x, kpt_y], dim=1)
    kpt_data = kpt_data.unsqueeze(0)
    
    with torch.no_grad():
        predict = model(kpt_data)
        
    return predict       


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam-steam', type=str, default='rtmp://...')
    parser.add_argument('--pose-engine', type=str, default='pretrained/yolov7pose_bs8.engine')
    parser.add_argument('--lstm-weight', type=str, default='weights/keypoints_lstm_v1.pt')
    parser.add_argument('--detect-sequence', type=int, default=45)

    args = parser.parse_args()
    
    video_path = args.video_path
    pose_engine = args.pose_engine
    lstm_weight = args.lstm_weight
    DETECT_SEQUENCE = args.detect_sequence
    
    stream = WebcamVideoStream("rtmp://live-10-hcm.fcam.vn:1956/168dda906d8b52b2?t=1675849366&tk=3dd50d058bed75717405eb9bff109c4451024a3d3eefc0de6c328ff83060e768/ViMkfBSt-OsytxdNQ-ZlXJqT0g-KQXhJBje-v2")
    
    print(vars(args))
    
    #  load yolopose tensorrt engine
    device = 'cuda:0'
    engine = TRTInferenceEngine(
        pose_engine,
        device
    )
    
    # load lstm model
    ckpt = torch.load(lstm_weight)
    params = ckpt['params']
    action_labels = ckpt['action_labels']
    
    lstm_model = KeypointsLSTM(num_features=params['num_features'], 
                               num_classes=params['num_classes'], 
                               lstm_layers=params['num_layers'], 
                               sequence_length=params['sequence_length'], 
                               hidden_dim=params['num_hidden_units'], 
                               device=device)
    
    lstm_model.load_state_dict(ckpt['weight'])
    lstm_model = lstm_model.eval().to(device)
    
    # init video reader
    img_provider = VideoReader(video_path)
   
    # init pose tracker
    tracker = PoseTracker()

    # inference
    t_start = time.time()
    visualize_imgs = []
    
    for batch_imgs in img_provider:
        
        batch_imgs, raw_imgs = preprocess_batch(batch_imgs)
        batch_imgs = batch_imgs.to(device)
        
        # pose estimation
        output_batch = engine(batch_imgs)[0]
        output_batch = nms_kpt(output_batch, conf_thres=0.15, iou_thres=0.5)
        
        for idx, op in enumerate(output_batch):
            if op.shape[0] == 0:
                continue
            
            #visualize
            raw_imgs[idx] = visualize_skeletons(raw_imgs[idx], op)
            
            # update pose tracker
            tracker.update(op)
            
            # checking pose sequence for fall detection    
            for t in tracker.online_targets:
                obj_id = int(t.track_id)
                objposes = tracker.pose_sequences[obj_id]

                if len(objposes) < DETECT_SEQUENCE:
                    print(f'object {obj_id} | poses length {len(objposes)}')
                    continue

                input_lstm = torch.cat(objposes, dim=0)
                predict = fall_detection(input_lstm, lstm_model)
                predict_class = action_labels[torch.argmax(predict)]
                
#                 import pdb;pdb.set_trace()
                # visualize
                cv2.putText(raw_imgs[idx], predict_class, t.tlwh[:2].astype(int), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                            thickness=3)
                
                # remove old pose from pose sequence
                tracker.pose_sequences[obj_id].remove(objposes[0])

                print(f'Fall detecting: object {obj_id} | predict {predict_class}')
                
            visualize_imgs.append(cv2.cvtColor(raw_imgs[idx], cv2.COLOR_BGR2RGB))
            # end batch
            
        # end video
                  
        t_end = time.time()        

        if t_end - t_start > 50:
            break
    
    # sumarize
    num_frames = img_provider.frame_cnt
    t_end = time.time()
    t_total = t_end - t_start
    
    print(f'Num frames: {num_frames}')
    print(f'Pipeline FPS: {num_frames/t_total}')
    
    # create video
    import moviepy.editor as mpy    
    vid = mpy.ImageSequenceClip(visualize_imgs, fps=15)
    video_name = video_path.split('/')[-1].split('.')[0]
    vid.write_videofile(f'demo/{video_name}_result.mp4')
    
    del engine
    