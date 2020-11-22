CUDA_VISIBLE_DEVICES=0 python train.py --input_wavs_dir ../../datasets/VCTK-Corpus-removed-silence/  --input_training_file ../../datasets/VCTK-Corpus-removed-silence/train_hifi-vctk-clean.txt   --input_validation_file ../../datasets/VCTK-Corpus-removed-silence/val_hifi-vctk-clean.txt --config config_v2.json    --checkpoint_path ../HiFi-Gan-checkpoints/not-emb/Finetuning/VCTK/Tacotron --fine_tuning True --input_mels_dir ../hifi-Gan-data/VCTK-Tacotron/mel/


