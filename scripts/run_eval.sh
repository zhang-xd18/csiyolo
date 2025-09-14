python ./tools/val.py \
    --Ns-max 10 \
    --imgsz 64 \
    --batch-size 64 \
    --data-path "/home/data/" \
    --device 0 \
    --project "/home/runs/" \
    --pretrained "/home/checkpoints/h5/ckpt.pt" \
    --cfg "/home/checkpoints/h5/config.yaml" \
    --detect-thres 1

#! NOTE:
# Ns-max - the maximum number of scatters for process.
# imgsz - the input image size for the model.
# batch-size - the batch size for testing.
# data-path - the path to the dataset.
# device - the GPU device to be used for testing.
# project - the directory to save the testing results.
# pretrained - the path to the pretrained model weights.
# cfg - the configuration file for the model, which includes the model architecture and other hyperparameters.
# detect-thres - the threshold for detecting a scatter, which is used to filter out false positives.

