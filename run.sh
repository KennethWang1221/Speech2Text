python3 ./whisper_split_gt.py\
        --model_type base.en\
        --root ./\
        --ext_txt .trans.txt\
        --ext_audio .flac\
        --folder_in_archive LibriSpeech\
        --url test-clean\
        --opts_dir ./res/whisper_test_clean_gt_input_list

python3 ./whisper_infer.py\
        --model_type base.en\
        --inputs_list ./res/whisper_test_clean_gt_input_list/inputs_list.txt\
        --opts_dir ./res/whisper_test_clean_opts

python3 ./whisper_wer_cal.py\
        --infer_opts_list ./res/whisper_test_clean_opts/opts_list.txt\
        --infer_inputs_list ./res/whisper_test_clean_gt_input_list/inputs_list.txt\
        --gt_dir ./res/whisper_test_clean_gt_input_list/gt_txt\
        --wer_save_dir ./res/whisper_test_clean_wer/wer_save_dir\