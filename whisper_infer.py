#!/usr/bin/env python3

import sys
import re
import argparse
import os
import numpy as np
import glob
import torch
import pandas as pd
import whisper
import torchaudio
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from pathlib import Path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_librispeech_metadata(audio_path, ext_txt='.trans.txt'):
    # Get audio path and sample rate
    fileid = os.path.splitext(os.path.basename(audio_path))[0]
    speaker_id, chapter_id, utterance_id = fileid.split("-")
    # Get audio path and sample rate
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"

    # Load gt text
    file_text = f"{speaker_id}-{chapter_id}{ext_txt}"
    file_text_root = audio_path.rsplit('/', 1)[0]
    file_text = os.path.join(file_text_root, file_text)

    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break

    metadata = {
        'filepath':audio_path,
        'SAMPLE_RATE':16000,
        'gt_path':file_text,
        'transcript':transcript,
        'speaker_id':int(speaker_id),
        'chapter_id':int(chapter_id),
        'utterance_id':int(utterance_id)
    }
    return metadata

def main(**args):
    opt_txt = os.path.join(args['opts_dir'], 'opt_txt')
    gt_txt = os.path.join(args['opts_dir'], 'gt_txt')
    opts_list = os.path.join(args['opts_dir'], 'opts_list.txt')

    if not os.path.exists(opt_txt):
        os.makedirs(opt_txt)

    if not os.path.exists(gt_txt):
        os.makedirs(gt_txt)

    model = whisper.load_model(args['model_type'])
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    with open(args['inputs_list'], "r") as f:
        walker = f.read().splitlines()
    
    opt_list = []
    for index, file in enumerate(walker):
        metadata = get_librispeech_metadata(file) 
        waveform, sample_rate = torchaudio.load(metadata['filepath'])
        audio = whisper.pad_or_trim(waveform.flatten()).to(DEVICE)
        mel = whisper.log_mel_spectrogram(audio)

        results = model.decode(mel, options)

        opt_txt_filename = os.path.join(opt_txt,metadata['filepath'].split('/')[-1].replace('.flac', '.txt'))
        gt_txt_filename = os.path.join(gt_txt,metadata['filepath'].split('/')[-1].replace('.flac', '.txt'))
        opt_list.append(opt_txt_filename)

        with open(opt_txt_filename, "w") as f1, open(gt_txt_filename, "w") as f2:
            f1.write(results.text)
            f2.write(metadata['transcript'])
            print("No : {}\ninput file: {}\nopts: {}\ntranscript: {}\n".format(index, metadata['filepath'],results.text,metadata['transcript']))
            print("="*10)

    with open(opts_list, "w") as f3:
        for filepath in opt_list:
            f3.write(filepath + "\n")

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Infer')
    # Load file
    parser.add_argument("--model_type", type=str, default="base.en", \
                        help='model type')
    parser.add_argument("--inputs_list", type=str,default="./res/whisper_test_clean_gt_input_list/inputs_list.txt", \
                        help='Path to the input list file')
    parser.add_argument('--opts_dir', type=str, default='./res/whisper_test_clean_opts',
                        help='path of outputs files')

    argspar = parser.parse_args()    

    print("\n### Test Whisper ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
