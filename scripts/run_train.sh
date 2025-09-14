python ./tools/train.py \
    --imgsz 64 \
    --batch-size 64 \
    --Ns-max 10 \
    --cfg "/home/csiyolo/configs/config.yaml" \
    --data-path "/home/data/" \
    --epochs 100 \
    --device 1 \
    --project "/home/runs/" \
    --noise-bound 0.0 \
    --detect-thres 1

#! NOTE:
# imgsz - the input image size for the model.
# batch-size - the batch size for training.
# Ns-max - the maximum number of scatters for process.
# cfg - the configuration file for the model, which includes the model architecture.
# data-path - the path to the dataset.
# epochs - the number of training epochs.
# device - the GPU device to be used for training.
# project - the directory to save the training results.
# noise-bound - the upper bound for the injected noise in training.
# detect-thres - the threshold for detecting a scatter, which is used to filter out false positives.



