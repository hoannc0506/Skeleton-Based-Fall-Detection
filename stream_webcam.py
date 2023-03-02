import os
import argparse
import threading
import time
from queue  import Queue
from threading import Thread
import cv2
import numpy as np
import paho.mqtt.publish as publish
# from flask  import Flask, Response, render_template
import logging
import warnings
import numpy as np
import subprocess as sp
import shlex
from datetime import datetime
import pycuda.driver as cuda
import pycuda.autoinit
from time import sleep
import torch
import cv2
import os
import glob
import time
import numpy as np
import onnxruntime as ort
import torchvision
from tracker.byte_tracker import BYTETracker
from collections import OrderedDict
import moviepy.editor as mpy 

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame        
        # from the stream        
        self.queue = Queue(maxsize=200)
        # self.stream = cv2.VideoCapture(src)        
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)        
        # (self.grabbed, self.frame) = self.stream.read()        
        self.frame = None        
        self.grabbed = False        
        # initialize the thread name        
        command = [ 'ffmpeg',
            # '-rtsp_transport', 'tcp',            
                    '-hwaccel', 'cuda',
                    '-c:v', 'h264_cuvid',
                    '-i', src,
                    '-pix_fmt', 'rgb24',  # brg24 for matching OpenCV            
                    # '-filter:v', 'fps=6',
                    '-f', 'rawvideo',
                    '-loglevel', 'error',
                    'pipe:' ]
        self.process = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8) 
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
            # (self.grabbed, self.frame) = self.stream.read()            
            buffer = self.process.stdout.read(self.W*self.H*3)
            if len(buffer) != self.W*self.H*3:
                break            
            img = np.frombuffer(buffer, np.uint8).reshape(self.H, self.W, 3)
            self.grabbed, self.frame = True, img            
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
        

## All pipeline in one file

###############################################################################################################
# Preprocess
def letterbox(im, new_shape=(768, 960), color=(114, 114, 114), auto=True, scaleup=True, stride=64):
    '''
    Pre process image for pose estimation
    '''
    
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def process_img(nimg, img_shape=(768,960)):
    '''
    Process cv2 image for onnx pose inference
    '''
    image, ratio, dwdh = letterbox(nimg, new_shape=img_shape, auto=False)
    raw_img = image.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    return im, raw_img


######################################################################################################
# Postprocess
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def output_to_kpt(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)


def nms_kpt(prediction, conf_thres=0.25, iou_thres=0.45, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,57) tensor per image [xyxy, conf, keypoints]
    """
    prediction = torch.from_numpy(prediction)

    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections

    t = time.time()
    output = [torch.zeros((0,6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        kpts = x[:, 6:]
        conf, j = x[:, 5:6].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
            
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def plot_skeleton_kpts(im, kpts, steps):
    #Plot the skeleton and keypoints for coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        
        if conf > 0.5:
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
        
def visualize_skeletons(nimg, skeletons, frame_idx):
    ''' Visualize skeletons for user inference
    nimg: cv2 image
    skeletons: detected skeletons
    frame_idx: frame index
    '''
    
    bboxes = xywh2xyxy(skeletons[:, 2:6])
    poses = skeletons[:, 7:]
    
    for idx in range(skeletons.shape[0]):
        plot_skeleton_kpts(nimg, poses[idx].T, 3)
        bbox = bboxes[idx].astype(int)
        cv2.rectangle(nimg,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      color=(255, 0, 0),
                      thickness=1,
                      lineType=cv2.LINE_AA)
        
    cv2.putText(nimg, f"{frame_idx}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    return nimg



###########################################################################################################
# Inference

class PoseEstimation():
    '''
        Inference yolov7 pose (onnx model) to extract keypoints 
    '''
    
    def __init__(self, model_path='pretrained/model.onnx', device='cuda'):
        self.model_path = model_path
        self.providers = ['CUDAExecutionProvider'] if device=='cuda' else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]
        self.stream = WebcamVideoStream("rtmp://live-10-hcm.fcam.vn:1956/168dda906d8b52b2?t=1675849366&tk=3dd50d058bed75717405eb9bff109c4451024a3d3eefc0de6c328ff83060e768/ViMkfBSt-OsytxdNQ-ZlXJqT0g-KQXhJBje-v2")
        self.warm_up()
        
        
        
    def warm_up(self):
        print("warm up.....")
        warmup_img = np.random.rand(1, 3, 768, 960).astype(np.float32)
        inp = {self.inname[0]:warmup_img}
        self.session.run(self.outname, inp)[0]
        self.stream.start()
        print("Done warmup")
        
    
    def inference(self, nimg): 
        input_img, raw_img = process_img(nimg)   
#         print(nimg.shape)
#         print(input_img.shape)
#         print(raw_img.shape)
        
        inp = {self.inname[0]:input_img} # input img shape [3, 768, 960]

        onnx_op = self.session.run(self.outname, inp)[0]
        outputs = nms_kpt(onnx_op, 0.25, 0.65, nc=1, nkpt=17)
        outputs = output_to_kpt(outputs)
    
        return outputs, raw_img

    
    
"""
Fall detection
"""
class FallDetection():
    '''
        Inference yolov7 pose (onnx model) to extract keypoints 
    '''
    
    def __init__(self, model_path='pretrained/model.onnx', device='cuda'):
        self.model_path = model_path
        
        self.providers = ['CUDAExecutionProvider'] if device=='cuda' else ['CPUExecutionProvider']

        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]
        
        self.warm_up()
        
        
    def warm_up(self):
        warmup_data = np.random.rand(1, 60, 38).astype(np.float32)
        inp = {self.inname[0]:warmup_data}
        self.session.run(self.outname, inp)[0]
        
    
    def inference(self, list_arr, img_shape=(768,960)): 
        arr_data = np.asarray(list_arr)
    
        arr_bbox = arr_data[:, 2:6]
        arr_kpt_x = arr_data[:, 7::3]
        arr_kpt_y = arr_data[:, 8::3]

        # center kpts data
        h,w = img_shape
        arr_bbox = arr_bbox / np.array([w,h,w,h])
        arr_kpt_x = (arr_kpt_x - 0.5*w) / (0.5*w)
        arr_kpt_y = (arr_kpt_y - 0.5*h) / (0.5*h)

        kpt_data = np.concatenate([arr_bbox, arr_kpt_x, arr_kpt_y], axis=1)
        kpt_data = np.expand_dims(kpt_data, 0).astype(np.float32)
        
        
        inp = {self.inname[0]:kpt_data}
        predict = self.session.run(self.outname, inp)[0]
        
        
        return predict 
    
    
    

if __name__== "__main__":
    
    pose_est = PoseEstimation('pretrained/yolov7-w6-pose_768_960_dynamic.onnx')
    fall_det = FallDetection('pretrained/keypoint_lstm.onnx')
    
    out_video = []
    frame_cnt = 0

    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.7, track_buffer=30, frame_rate=30)
    img_size = (768,960)

    out_video = []
    # tracking pose
    pose_sequences = OrderedDict()
    DETECT_SEQUENCE = 60
    label_list = ['fall', 'normal']
    
    try:
       
        while True:
            data, _ = pose_est.stream.read()

            ret, frame = data
            if not ret:
                time.sleep(0.1)
                continue  
                
            frame_cnt +=1
            
            outputs, raw_img = pose_est.inference(frame)

            if outputs.shape[0]==0:
                out_video.append(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
                continue
            
            outputs = outputs.reshape(-1, 58)
            online_targets = tracker.update(outputs, img_size, img_size)

            # remove loss object
            for rm_obj in tracker.removed_stracks:
                rm_id = rm_obj.track_id
                if rm_id in pose_sequences:
                    del pose_sequences[rm_id]
              
            # tracking object and predict
            num_p = 0
            for t in online_targets:
                num_p +=1
                # check exist id     

                tid = t.track_id  
                obj_id = int(tid)
                score = t.score                         
                pose = t.pose     
                centroid = pose[2:4].astype(int)
#                 print(pose.shape)
                drawbox = xywh2xyxy(pose[2:6].reshape(1,-1)).astype(int)[0] # remove unuse shape 0

                if obj_id not in pose_sequences:
                    pose_sequences[obj_id] = [] 
                else:
                    pose_sequences[obj_id].append(pose)

                our_img = visualize_skeletons(raw_img, pose.reshape(1,-1), frame_cnt)  

                cv2.putText(raw_img, f'{obj_id}', (centroid[0], centroid[1]), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),
                            thickness=3)
                objposes = pose_sequences[obj_id]

                if len(objposes) >= DETECT_SEQUENCE:
                    det_t_start = time.time()

                    predict = fall_det.inference(objposes)
                    predict_class = label_list[np.argmax(predict)]
                    # remove old pose from pose sequence
                    pose_sequences[obj_id].remove(objposes[0])

                    cv2.putText(our_img, predict_class, (drawbox[0], drawbox[1]), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),
                                thickness=3)
                    print(f'Fall detecting: frame {frame_cnt} object {obj_id} - {predict_class} {predict}')

#                     total_det_time += time.time() - det_t_start  

            print(f'Num person: {num_p}')

            out_video.append(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
#             t_end = time.time()        

#             if t_end - t_start > 50:
#                 break
            
    except KeyboardInterrupt:   
        vid = mpy.ImageSequenceClip(out_video, fps=20)
        vid.write_videofile('demo_pose_estimation.mp4')
        


        
