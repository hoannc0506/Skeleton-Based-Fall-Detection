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
import threading
from queue import Queue
from threading import Thread
from flask import Flask, Response, render_template


from collections import OrderedDict
from models.lstm_models import KeypointsLSTM
from models.tensorrt_inference import TRTInferenceEngine

from utils.general import *
from utils.stream_webcam import WebcamVideoStream
from utils.inference_utils import *

from tracker.byte_tracker import BYTETracker


outputFrame = None
# lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def display():
    # return the rendered template
    return render_template("index.html")

def generate():
    global outputFrame
    
    while True:
        if outputFrame is None:
            continue
        
        frame = outputFrame
        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')



@app.route("/video_feed_raw")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    print('asdf')
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
     


def main(args_dict):
    global outputFrame
    
    args = argparse.Namespace(**args_dict)
    pose_engine = args.pose_engine
    lstm_weight = args.lstm_weight
    img_shape = args.img_size
    is_visualize = args.visualize
    grab_frame = args.grab_frame
    MAX_LENGTH = args.max_length
    MIN_LENGTH = args.min_length
    
    print(vars(args))
    
    stream = WebcamVideoStream('rtmp://live-10-hcm.fcam.vn:1956/168dda906d8b52b2?t=1675849366&tk=3dd50d058bed75717405eb9bff109c4451024a3d3eefc0de6c328ff83060e768/ViMkfBSt-OsytxdNQ-ZlXJqT0g-KQXhJBje-v2')
    stream.start()
    
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
    tracker = PoseTracker(track_thresh=0.5, match_thresh=0.8, img_shape=img_shape)

    # inference
    t_start = time.time()
    
    batch_size = 8
    batch_cnt = 0
    frame_cnt = 0
    batch_list=[]
    
    # try:
    while True:
    
        data, _ = stream.read()
        
        ret, frame = data
        
        if not ret:
            time.sleep(0.1)
            continue  
        
        batch_list.append(frame)
        frame_cnt +=1
        batch_cnt +=1

        if batch_cnt < batch_size:
            continue

        batch_imgs, raw_imgs = preprocess_batch(batch_list, out_img_shape=img_shape)
        batch_imgs = batch_imgs.to(device)
        
        # pose estimation
        output_batch = engine(batch_imgs)[0]
        output_batch = nms_kpt(output_batch, conf_thres=0.15, iou_thres=0.5)
        
        for idx, op in enumerate(output_batch):
            if op.shape[0] == 0:
                outputFrame = letterbox(raw_imgs[idx], (1080,1920), stride=64)[0].copy()
                continue
            
            #visualize
            raw_imgs[idx] = visualize_skeletons(raw_imgs[idx], op)
            
            # update pose tracker
            tracker.update(op)
            
            # checking pose sequence for fall detection    
            for t in tracker.online_targets:
                obj_id = int(t.track_id)
                objposes = tracker.pose_sequences[obj_id]
                
                # visualize
                cv2.putText(raw_imgs[idx], str(obj_id), t.mean[:2].astype(int),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 51, 255),
                            thickness=3)

                if len(objposes) < MIN_LENGTH:
                    print(f'object {obj_id} | poses length {len(objposes)}')
                    continue

                input_lstm = torch.cat(objposes, dim=0)
                
                # detect fall
                predict = detect_fall(input_lstm, lstm_model, img_shape=img_shape)
                
                predict_class = action_labels[torch.argmax(predict)]
                
                # visualize
                color = (0, 0, 255) if predict_class=='fall' else (0, 255, 0)
                cv2.putText(raw_imgs[idx], predict_class, t.tlwh[:2].astype(int),
                            cv2.FONT_HERSHEY_PLAIN, 2, color,
                            thickness=3)
                
                # remove old pose from pose sequence
                if len(objposes) >= MAX_LENGTH:
                    tracker.pose_sequences[obj_id].remove(objposes[0])

                print(f'Fall detecting: object {obj_id} | predict {predict_class} {torch.max(predict)}')
                
            outputFrame = letterbox(raw_imgs[idx], (1080,1920), stride=64)[0].copy()
            # end batch

        batch_list = []    
    del engine
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--grab-frame', type=int, default=1)
    parser.add_argument('--pose-engine', type=str, default='pretrained/yolov7-w6-pose-960.engine')
    parser.add_argument('--lstm-weight', type=str, default='weights/kpt_lstm_960_all_data.pt')
    parser.add_argument('--img-size', nargs='+', type=int, default=(960, 960))  
    parser.add_argument('--min-length', type=int, default=30)
    parser.add_argument('--max-length', type=int, default=45)
    parser.add_argument('--visualize', type=bool, default=False)

    args = parser.parse_args()
    
    Thread(target=main, args=(vars(args),), daemon=True).start()
    app.run(host='0.0.0.0', port=9095, debug=True,
            threaded=True, use_reloader=False)
    
    
    
    
    
    
    
    

    