import cv2
import os
import torch
import time
import numpy as np
import argparse

from tqdm import tqdm
from lstm_models import KeypointsLSTM
from pose_utils import *

from tracker.byte_tracker import BYTETracker
from collections import OrderedDict
 

class VideoReader(object):
    '''
        Read .mp4 video files
    '''
    def __init__(self, file_name):
        self.file_name = file_name
        self.frame_cnt = 0

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError(f'Video {self.file_name} cannot be opened')
        return self

    def __next__(self):
        # for i in range(3):
        #     self.cap.grab()
            
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        self.frame_cnt += 1
        return img
    

def fall_detection(list_arr, model, img_shape=(768, 960)):
    '''
    input: list tensor arr of keypoint data of 1 person, model 
    output: is falling detected
    '''
    # preprocess
    arr_bbox = arr_data[:, 2:6]
    arr_kpt_x = arr_data[:, 7::3]
    arr_kpt_y = arr_data[:, 8::3]
    
    # center kpts data
    h,w = img_shape
    arr_bbox = arr_bbox / np.array([w,h,w,h])
    arr_kpt_x = (arr_kpt_x - 0.5*w) / (0.5*w)
    arr_kpt_y = (arr_kpt_y - 0.5*h) / (0.5*h)
    
    kpt_data = np.concatenate([arr_bbox, arr_kpt_x, arr_kpt_y], axis=1)
    
    kpt_data = torch.from_numpy(kpt_data).unsqueeze(0)
    kpt_data = kpt_data.to(device).float()

    with torch.no_grad():
        predict = model(kpt_data)
        
    return predict


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='video_demo/demo_fall_cam1.mp4')
    parser.add_argument('--pose-engine', type=str, default='weight/yolov7-w6-pose.pt')
    parser.add_argument('--lstm-weight', type=str, default='weight/keypoints_lstm_v1.pth')
    parser.add_argument('--detect-sequence', type=int, default=30)

    args = parser.parse_args()
    
    video_path = args.video_path
    pose_weight = args.pose_engine
    lstm_weight = args.lstm_weight
    DETECT_SEQUENCE = args.detect_sequence
    
    print(vars(args))

    device = torch.device('cuda')
    
    #  load yolopose model
    weigths = torch.load(pose_weight, map_location=device)
    pose_model = weigths['model']
    _ = pose_model.float().eval()
    _ = pose_model.half().to(device)
    
    # load lstm model
    ckpt = torch.load(lstm_weight)
    
    model_params = ckpt['params']
    lstm_model = KeypointsLSTM(num_features=38, 
                  num_classes=2, 
                  lstm_layers=model_params['num_layers'], 
                  sequence_length=model_params['sequence_length'], 
                  hidden_dim=model_params['num_hidden_units'], 
                  device=device)
    
    lstm_model.load_state_dict(ckpt['weight'])
    lstm_model = lstm_model.eval().to(device)
    
    
    img_provider = VideoReader(video_path)
    
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.7, track_buffer=30, frame_rate=30)
    img_size = (768,960)
    
    visualize_imgs = []
    action_labels = ['fall', 'normal']
    
    # tracking pose
    pose_sequences = OrderedDict()
    t_start = time.time()
    total_det_time = 0
    total_est_time = 0
    
    
    for img in img_provider:
        est_time = time.time()
        outputs, raw_img = get_skeletons(img, pose_model)   
        total_est_time += time.time() - est_time
        
        if outputs is None:
            visualize_imgs.append(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
            continue
        
        online_targets = tracker.update(outputs, img_size, img_size)
        
        # remove loss object
        for rm_obj in tracker.removed_stracks:
            rm_id = rm_obj.track_id
            if rm_id in pose_sequences:
                del pose_sequences[rm_id]
            
                
        for t in online_targets:
            # check exist id     
          
            tid = t.track_id  
            obj_id = int(tid)
            score = t.score                         
            pose = t.pose     

            drawbox = xywh2xyxy(pose[2:6].reshape(1,-1)).astype(int)[0] # remove unuse shape 0

            if obj_id not in pose_sequences:
                pose_sequences[obj_id] = [] 
            else:
                pose_sequences[obj_id].append(pose)

            cv2.rectangle(raw_img, drawbox[0:2], drawbox[2:4], 
                          color=(255, 0, 0), thickness=2)

            cv2.putText(raw_img, f'{obj_id}', pose[2:4].astype(int), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),
                        thickness=3)

            plot_skeleton_kpts(raw_img, pose[7:].T, 3) 
            objposes = pose_sequences[obj_id]

            if len(objposes) >= DETECT_SEQUENCE:
                det_t_start = time.time()

                predict = fall_detection(objposes, lstm_model, device)
                predict_class = action_labels[torch.argmax(predict)]
                
                # remove old pose from pose sequence
                pose_sequences[obj_id].remove(objposes[0])
                
                cv2.putText(raw_img, predict_class, (drawbox[0], drawbox[1]), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                            thickness=3)
                print(f'Fall detecting: frame {img_provider.frame_cnt} \
                        object {obj_id} - predict {predict_class}')
                
                total_det_time += time.time() - det_t_start                                    

        visualize_imgs.append(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        t_end = time.time()        

        if t_end - t_start > 50:
            break
    
    print(f'Process time: {t_end-t_start}')
    print(f'Fall detection time: {total_det_time}')
    print(f'Pose estimation time: {total_est_time}')
    print(f'Num frames: {img_provider.frame_cnt}')
    print(f'Pipeline FPS: {img_provider.frame_cnt/(t_end-t_start)}')
    print(f'Pose estimation FPS: {img_provider.frame_cnt/total_est_time}')
    
    import moviepy.editor as mpy    
    vid = mpy.ImageSequenceClip(visualize_imgs, fps=20)
    video_name = video_path.split('/')[-1].split('.')[0]
    vid.write_videofile(f'demo/{video_name}_result.mp4')