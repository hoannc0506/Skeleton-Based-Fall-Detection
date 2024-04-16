# Fall detection

## Pipeline
![pipeline](./results/pipeline.png)
+ This repo is my implementation code for my first [paper](https://conferences.hcmut.edu.vn/conference/paper/64280fea81914a42d49c4ab8/abstractFile/pd_iJrxb2dkQYCb7Ml6F8_WG.pdf)

## Performance
+ FPS: 
    + Video ~ 30-35FPS
    + Stream ~ 20-22FPS
+ Accuracy:
    + Precision: 96%
    + Recall: 90%

## Uses
+ Download [LSTM_classifier](https://drive.google.com/file/d/1S5hQdp-OuaLP28khi6IWIEpXs_HayZPW/view?usp=sharing) and put  it in folder `weights`
+ Download yolov7-pose estimation pretrained from [YOLOv7 Repo](https://github.com/WongKinYiu/yolov7)
+ Build TensorRT engine

```
CUDA_VISIBLE_DEVICES=0 python models/tensorrt_builder.py \
--onnx-path pretrained/yolov7-w6-pose.onnx --onnx-input batch_images \
--save-path pretrained/yolov7-w6-pose-960.engine \
--min-bs 1 --max-bs 8 --optim-bs 8 \
--img-height 960 --img-width 960
```

+ Inference video

```
CUDA_VISIBLE_DEVICES=0 python inference_video.py \
--pose-engine pretrained/yolov7-w6-pose-960.engine \
--video-path ../dataset/cauca/fall/FallForwardS5.avi \
--grab-frame 1 --img-height 960 --img-width 960 \
--detect-sequence 30 --visualize 

```

## Further Improvement
At the time when I implemented this methodology, YOLOv8 and YOLOv9 had not yet been released, leaving ample room for improvement. These areas can be enhanced:
- Enhance the pose estimation model
- Refine the tracking method
- Develop a more effective training strategy
- Increase the volume of available data
