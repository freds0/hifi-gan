CUDA_VISIBLE_DEVICES=0 python train.py \
    --input_wavs_dir ./LJSpeech-1.1/wavs  \
    --input_training_file ./LJSpeech-1.1/training.txt   \
    --input_validation_file ./LJSpeech-1.1/validation.txt \
    --config config_v1.json   \
    --checkpoint_path ./checkpoints
