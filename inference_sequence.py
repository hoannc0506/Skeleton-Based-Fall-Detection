# import trt befor torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import cv2
import os
import time
import glob
import numpy as np
import argparse
import logging

from collections import OrderedDict
from models.lstm_models import KeypointsLSTM
from models.tensorrt_inference import TRTInferenceEngine

from utils.general import *
from utils.stream_webcam import WebcamVideoStream

from tracker.byte_tracker import BYTETracker


class SequenceReader(object):
    '''
        Read sequence images
    '''

    def __init__(self, folder_path, grab_frame=1, batch_size=8):
        self.folder_path = folder_path
        if 'harup' in folder_path:
            self.file_names = sorted(glob.glob(folder_path+'/*.jpg'), key=lambda x: len(x))
#             import pdb
        else:
            self.file_names = sorted(glob.glob(folder_path+'/*.jpg'))
            
        self.max_frame = len(self.file_names)
        self.batch_size = batch_size
        self.grab_frame = grab_frame
        self.batch_cnt = 0
        print(self.max_frame)
        print(self.file_names[:5])

    def __iter__(self):
        self.frame_cnt = 0
        return self

    def __next__(self):
        if self.frame_cnt >= self.max_frame:
            raise StopIteration
                
        batch_imgs = []

        for i in range(self.batch_size):
            # grab 1 frame
            self.frame_cnt += self.grab_frame
            if self.frame_cnt >= self.max_frame:
                raise StopIteration
            
#             print(self.frame_cnt)
            img = cv2.imread(self.file_names[self.frame_cnt])
            
            if img.shape == 0:
                raise IOError('Image {} cannot be read'.format(self.file_names[self.frame_cnt]))

            batch_imgs.append(img)
            self.frame_cnt += 1
        
        
        self.batch_cnt += 1
        return batch_imgs


class PoseTracker():
    '''
        Object tracking for fall detection
    '''
    def __init__(self, track_thresh=0.6, match_thresh=0.6, track_buffer=30, 
                       frame_rate=15, img_shape=(768,960)):
        
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


def detect_fall(kpt_data, model, img_shape=(768, 960)):
    '''
        input 
            kpt_data: tensor keypoints data of 1 person shape [num_frame, 38]
            model: keypoint_lstm model
            
        output: 
            predict
    '''
    # preprocess
    bbox = kpt_data[:, :4]
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


def inference(img_provider, engine, lstm_model, tracker, 
              device='cuda:0', img_shape=(768,960), 
              min_length=30, max_length=45, 
              labels=None, visualize=False):
    
    # inference
    t_start = time.time()
    visualize_imgs = []
    predicts = []
    no_background = False
    
    for batch_imgs in img_provider:

        batch_imgs, raw_imgs = preprocess_batch(batch_imgs, out_img_shape=img_shape)
        batch_imgs = batch_imgs.to(device)

        # pose estimation
        output_batch = engine(batch_imgs)[0]
        output_batch = nms_kpt(output_batch, conf_thres=0.15, iou_thres=0.5)
        
        for idx, op in enumerate(output_batch):
            if op.shape[0] == 0:
                continue
                
            vis_img = raw_imgs[idx].copy()
            if no_background:
                vis_img[...] = 114
                
            #visualize
            vis_img = visualize_skeletons(vis_img, op)
            
            # update pose tracker
            tracker.update(op)
            
            # checking pose sequence for fall detection    
            for t in tracker.online_targets:
#                 import pdb; pdb.set_trace()
                
                obj_id = int(t.track_id)
                objposes = tracker.pose_sequences[obj_id]
                
                # visualize
                cv2.putText(raw_imgs[idx], str(obj_id), t.mean[:2].astype(int),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 51, 255),
                            thickness=3)

                if len(objposes) < min_length:
                    print(f'object {obj_id} | poses length {len(objposes)}')
                    continue

                input_lstm = torch.cat(objposes, dim=0)
                
                # detect fall
                predict = detect_fall(input_lstm, lstm_model, img_shape=img_shape)
                
                predict_class = labels[torch.argmax(predict)]
                predicts.append(torch.argmax(predict))
                
                # visualize
                color = (0, 0, 255) if predict_class=='fall' else (0, 255, 0)
                cv2.putText(raw_imgs[idx], predict_class, t.tlwh[:2].astype(int),
                            cv2.FONT_HERSHEY_PLAIN, 3, color,
                            thickness=3)
                
                # remove old pose from pose sequence
                if len(objposes) >= max_length: 
                    tracker.pose_sequences[obj_id].remove(objposes[0])

                print(f'Fall detecting: object {obj_id} | predict {predict_class} {torch.max(predict)}')
                
            visualize_imgs.append(cv2.cvtColor(raw_imgs[idx], cv2.COLOR_BGR2RGB))
            # end batch
            
        # end video
                  
        t_end = time.time()
        if t_end - t_start > 50:
            break
    
    # sumarize
    num_frames = img_provider.frame_cnt
    num_batchs = img_provider.batch_cnt
    batch_size = img_provider.batch_size
    t_end = time.time()
    t_total = t_end - t_start
    
    print(f'Num frames: {num_frames}')
    print(f'Num batchs: {num_batchs}')
    print(f'Pipeline FPS: {(num_batchs*batch_size)/t_total}')
    
    if visualize:
    # create video
        import moviepy.editor as mpy    
        vid = mpy.ImageSequenceClip(visualize_imgs, fps=20)
        video_name = img_provider.folder_path.split('/')[-1]
        save_path = f'results/{video_name}_result.mp4'
        vid.write_videofile(save_path)
    
    
    return predicts
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-path', type=str, default='demo/fall2.mp4')
    parser.add_argument('--grab-frame', type=int, default=0)
    parser.add_argument('--pose-engine', type=str, default='pretrained/yolov7-w6-pose-960.engine')
    parser.add_argument('--lstm-weight', type=str, default='weights/kpt_lstm_960.pt')
    parser.add_argument('--img-size', nargs='+', type=int, default=[960, 960])  
    parser.add_argument('--min-length', type=int, default=30)
    parser.add_argument('--max-length', type=int, default=45)

    parser.add_argument('--visualize', action='store_true')
    
    args = parser.parse_args()
    
    sequence_path = args.sequence_path
    pose_engine = args.pose_engine
    lstm_weight = args.lstm_weight
    img_shape = args.img_size
    is_visualize = args.visualize
    grab_frame = args.grab_frame
    MIN_LENGTH = args.min_length
    MAX_LENGTH = args.max_length
    
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
    action_labels = ['fall', 'normal']
    
    lstm_model = KeypointsLSTM(num_features=params['num_features'], 
                               num_classes=params['num_classes'], 
                               num_layers=params['num_layers'], 
                               sequence_length=params['sequence_length'], 
                               hidden_dim=params['num_hidden_units'], 
                               device=device)
    
    lstm_model.load_state_dict(ckpt['weight'])
    lstm_model = lstm_model.eval().to(device)
    
    # init pose tracker
    posetracker = PoseTracker(track_thresh=0.5, match_thresh=0.7, img_shape=img_shape)
    
    # init sequence reader
    sequence_reader = SequenceReader(sequence_path, grab_frame=grab_frame)
    
    # inference
    output = inference(sequence_reader, engine, lstm_model, 
                       posetracker, device, img_shape,
                       MIN_LENGTH, MAX_LENGTH,
                       action_labels, is_visualize)
    
    del engine