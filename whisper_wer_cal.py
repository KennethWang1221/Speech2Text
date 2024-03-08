#!/usr/bin/env python3
import os
import numpy as np
import argparse

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import torch
import pandas as pd
from pathlib import Path
import glob
from jiwer import compute_measures
from whisper.normalizers import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from itertools import chain

def word2char(reference, hypothesis):
    # tokenize each word into an integer
    vocabulary = set(chain(*reference, *hypothesis))

    if "" in vocabulary:
        raise ValueError(
            "Empty strings cannot be a word. "
            "Please ensure that the given transform removes empty strings."
        )

    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    reference_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in reference
    ]
    hypothesis_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in hypothesis
    ]

    return reference_chars, hypothesis_chars

def main(**args):

    csv_file = os.path.join(args['wer_save_dir'],'wer_res.csv')

    if not os.path.exists(args['wer_save_dir']):
        os.makedirs(args['wer_save_dir'])

    if not os.path.exists(args['infer_inputs_list']):
        raise ValueError(f"Invalid list `{os.path.abspath(args['infer_inputs_list'])}`")

    if not os.path.exists(args['infer_opts_list']):
        raise ValueError(f"Invalid list `{os.path.abspath(args['infer_opts_list'])}`")

    if not os.path.exists(args['gt_dir']):
        raise ValueError(f"Invalid folder `{os.path.abspath(args['gt_dir'])}`")
    else:
        gts_list = sorted(glob.glob(args['gt_dir'] + '/*.txt'))
        gts_count = len(gts_list)

    with open(args['infer_opts_list'], 'r') as opts_list_file:
        opts_list = opts_list_file.readlines()
        opts_count = len(opts_list)

    with open(args['infer_inputs_list'], 'r') as inputs_list_file:
        inputs_list = inputs_list_file.readlines()
        inputs_count = len(inputs_list)

    if opts_count != inputs_count or inputs_count != gts_count:
        raise ValueError(f"The number of files in infer_opts_list ({opts_count}) does not match the number of files in infer_inputs_list ({inputs_count}).")
    else:
        print("{} files in {}\n{} files in {}\n{} files in {}".format(inputs_count, args['infer_inputs_list'], opts_count, args['infer_opts_list'],gts_count,args['gt_dir']))

    num_substitutions, num_deletions, num_insertions, num_hits,num_rf_words_list, num_hp_words_list = [], [], [], [], [], []

    gt_list = []
    infer_list = []
    for index, infer_file in enumerate(opts_list):
        gt_file = os.path.join(gts_list[index].rsplit('/',1)[0],infer_file.strip().rsplit('/',1)[1])
        if not os.path.exists(gt_file):
            raise ValueError(f"Invalid file `{os.path.abspath(gt_file)}`")
        else:
            with open(gt_file, "r") as f1, open(infer_file.strip(), "r") as f2:
                gt = f1.read().splitlines()
                infer = f2.read().splitlines()
                gt_list.extend(gt)
                infer_list.extend(infer)
            gt_clean = [normalizer(text) for text in gt]
            infer_clean = [normalizer(text) for text in infer]
            error_dict = compute_measures(gt_clean, infer_clean)

            num_substitutions.append(error_dict['substitutions'])
            num_deletions.append(error_dict['deletions'])
            num_insertions.append(error_dict['insertions'])
            num_hits.append(error_dict['hits'])

            ref_as_chars, hyp_as_chars = word2char(error_dict['truth'], error_dict['hypothesis'])
            num_rf_words_list.append(len(ref_as_chars[0]))
            num_hp_words_list.append(len(hyp_as_chars[0])) 
            
            if 'data' not in locals():
                data = pd.DataFrame(columns=["audio_path", "opt_path", "hypothesis", "reference", "Substitutions", "Deletions", "Insertions", "Hits", "WER", "MER", "WIL", "WIP"])
            temp_df = pd.DataFrame({
                "audio_path":inputs_list[index].strip(),
                "opt_path":infer_file.strip(),
                "hypothesis": infer_clean,
                "reference": gt_clean,
                "Substitutions": error_dict['substitutions'],
                "Deletions": error_dict['deletions'],
                "Insertions": error_dict['insertions'],
                "Hits": error_dict['hits'],
                "WER": error_dict['wer'],
                "MER":error_dict['mer'],
                "WIL":error_dict['wil'],
                "WIP":error_dict['wip']
            })
            data = pd.concat([data, temp_df], ignore_index=True)

    S, D, I, H, num_rf_words, num_hp_words = sum(num_substitutions), sum(num_deletions), sum(num_insertions), sum(num_hits), sum(num_rf_words_list), sum(num_hp_words_list)
    wer_total = float(S + D + I) / float(H + S + D)
    mer_total = float(S + D + I) / float(H + S + D + I)
    wip_total = (
        (float(H) / num_rf_words) * (float(H) / num_hp_words)
        if num_hp_words >= 1
        else 0
    )
    wil_total = 1 - wip_total

    total_summary_row = pd.DataFrame({
        "audio_path":["TOTAL_AUDIO"],
        "opt_path":["TOTAL_AUDIO"],
        "hypothesis": ["TOTAL_AUDIO"],
        "reference": ["TOTAL_AUDIO"],
        "Substitutions": [S],
        "Deletions": [D],
        "Insertions": [I],
        "Hits": [H],
        "WER": [wer_total],
        "MER":[mer_total],
        "WIL":[wil_total],
        "WIP":[wip_total]
    })

    data = pd.concat([data, total_summary_row], ignore_index=True)
    data.to_csv(csv_file, index=False)

    print("Total Files: {}\nnum_substitutions: {}\nnum_deletions:{}\nnum_insertions:{}\nnum_hits:{}\nWER_total: {}\nMER_total: {}\nWIL_total: {}\nWIP_total : {}\n"
        .format(index+1,S,D,I,H,wer_total,mer_total,wil_total,wip_total))

    
if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval WER')
    # Load file
    parser.add_argument("--infer_opts_list", type=str, default="./res/whisper_test_clean_opts/opts_list.txt", \
                        help='Path to outputs list text file')
    parser.add_argument("--infer_inputs_list", type=str,default="./res/whisper_test_clean_gt_input_list/inputs_list.txt", \
                        help='Path to original audio inputs list text file')
    parser.add_argument('--gt_dir', type=str, default='./res/whisper_test_clean_gt_input_list/gt_txt',
                        help='Path to the ground truth dir')
    parser.add_argument('--wer_save_dir', type=str, default='./res/whisper_test_clean_wer/wer_save_dir',
                        help='Path to wer result save dir')
    
    argspar = parser.parse_args()    

    print("\n### Eval Whisper WER ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
