CUDA_VISIBLE_DEVICES=6 python3 train.py --input_wavs_dir  /mnt/fred/tacotron2-GS/BRSpeech/  --input_training_file /mnt/fred/tacotron2-GS/BRSpeech/train_hifi-44khz-clean.txt   --input_validation_file /mnt/fred/tacotron2-GS/BRSpeech/val_hifi-44khz-clean.txt --config config_v1.json   --checkpoint_path ../checkpoints/HiFi-GAN-upsample-interpolation/v1/GS-TL-Tacotron-Nvidia/ --fine_tuning True --input_mels_dir /mnt/fred/tacotron2-GS/BRSpeech/GS_MelSpecs_HiFi-GAN/
