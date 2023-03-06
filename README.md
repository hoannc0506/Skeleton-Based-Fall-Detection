# Fall detection

## Pipeline
![pipeline](./results/pipeline.png)
+ Video/stream -> Pose estimation(TensorRT) -> Skeletons -> Preprocess -> LSTM classifier -> Normal/Fall

## Performance
+ FPS: 
    + Video ~ 30-35FPS
    + Stream ~ 20-22FPS
+ Accuracy:
    + Precision: 96%
    + Recall: 90%

## Run script
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

