# train
CUDA_VISIBLE_DEVICES=3 python train_v2/evaluate_dataset.py \
--pose-engine pretrained/yolov7-w6-pose-960.engine \
--lstm-weight weights/kpt_lstm_960.pt \
--img-size 960 960 \
--summary-path summary_failcase.txt \
--visualize

# evaluate model
CUDA_VISIBLE_DEVICES=3 python train_v2/evaluate_dataset.py \
--pose-engine pretrained/yolov7-w6-pose-960.engine \
--lstm-weight weights/kpt_lstm_960.pt \
--img-size 960 960 \
--summary-path summary_failcase.txt \
--visualize

