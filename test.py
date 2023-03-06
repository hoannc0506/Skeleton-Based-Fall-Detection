import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import cv2
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from models.lstm_models import KeypointsLSTM
from models.tensorrt_inference import TRTInferenceEngine

from inference_video import PoseTracker, VideoReader, inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='dataset')
    parser.add_argument('--grab-frame', type=int, default=1)
    parser.add_argument('--pose-engine', type=str, default='pretrained/yolov5-l6-pose-bs8.engine')
    parser.add_argument('--lstm-weight', type=str, default='weights/keypoints_lstm_v1.pt')
    parser.add_argument('--img-height', type=int, default=768)
    parser.add_argument('--img-width', type=int, default=960)    
    parser.add_argument('--detect-sequence', type=int, default=45)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    pose_engine = args.pose_engine
    lstm_weight = args.lstm_weight
    img_shape = (args.img_height, args.img_width)
    is_visualize = args.visualize
    grab_frame = args.grab_frame
    DETECT_SEQUENCE = args.detect_sequence
    
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
    
    video_paths = glob.glob(dataset_path+ '/*/*.avi')
    print(len(video_paths))
          
    predicts = []
    labels = []
    
    for video_path in tqdm(video_paths):
        # init pose tracker
        tracker = PoseTracker(track_thresh=0.5, match_thresh=0.7, img_shape=img_shape)
        
        # init video reader
        video_reader = VideoReader(video_path, grab_frame=grab_frame)

        # inference
        output = inference(video_reader, engine, lstm_model, tracker, device, 
                           img_shape, sequence_length=DETECT_SEQUENCE, labels=action_labels)
        
        if 0 in output:
            predicts.append('fall')
        else:
            predicts.append('normal')
        
        if 'Fall' in video_path:
            labels.append('fall')
        else:
            labels.append('normal')
            
        del tracker
            
    
    print(f'Model accuracy score: {accuracy_score(labels, predicts):0.4f}')
    print(classification_report(labels, predicts))
                        
    del engine
    