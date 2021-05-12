CUDA_VISIBLE_DEVICES=5 python train.py --input_wavs_dir ../../../lucasgris/wav2vec2/Datasets/TED_16K_unlabeled_V1/ --input_training_file ../../../lucasgris/wav2vec2/Datasets/TED_16K_unlabeled_V1/train_hifi-clean.txt   --input_validation_file ../../../lucasgris/wav2vec2/Datasets/TED_16K_unlabeled_V1/val_hifi-clean.txt --config config_v1_16khz.json   --checkpoint_path ../checkpoints/HiFi-GAN-upsample-interpolation/v1/TED/


CUDA_VISIBLE_DEVICES=6 python get-best-checkpoint.py  --input_wavs_dir ../../../lucasgris/wav2vec2/Datasets/CV_V2/ --checkpoint_path ../checkpoints/HiFi-GAN-upsample-interpolation/v1/TED/ --input_validation_file  ../../../lucasgris/wav2vec2/Datasets/CV_V2/eval-upsample-clean.txt --config config_v1_16khz.json --batch_size 20
