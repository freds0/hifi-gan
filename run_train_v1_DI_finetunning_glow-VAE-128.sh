CUDA_VISIBLE_DEVICES=3 python3.6 train.py --input_wavs_dir /mnt/fred/BRSpeech-Datasets/TTS/BRSpeech-TTS-3.0-beta7/  --input_training_file /mnt/fred/tacotron2-DI/BRSpeech/train_hifi-Brspeech-DI.txt   --input_validation_file /mnt/fred/tacotron2-DI/BRSpeech/val_hifi-Brspeech-DI.txt  --config config_v1.json    --checkpoint_path ../checkpoints/HiFi-GAN-upsample-interpolation/v1/DI-TL-SC_GlowTTS_VAE_128/ --fine_tuning True --input_mels_dir ../hifi-Gan-data/TL-Portuguese-DI-without-noise_scale/ --training_epochs 100000


 