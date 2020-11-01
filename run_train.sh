CUDA_VISIBLE_DEVICES=2 python train.py --input_wavs_dir ../../datasets/LibriTTS/LibriTTS/dataset-preprocessed-clean-100-and-360/ --input_training_file ../../datasets/LibriTTS/LibriTTS/dataset-preprocessed-clean-100-and-360/train_hifi-libritts-clean.txt   --input_validation_file ../../datasets/LibriTTS/LibriTTS/dataset-preprocessed-clean-100-and-360/val_hifi-libritts-clean.txt --config config_v2.json   --checkpoint_path ../HiFi-Gan-checkpoints/not-emb/LibriTTS/
 
 #--speakers_json ../hifi-Gan-data/speakers_test.json  

