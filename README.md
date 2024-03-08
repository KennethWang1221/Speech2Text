# Speech2Text Based on Whisper

This repository provides a fast inference API to complete Speech2Text task.



## Model

The base.en from OpenAI is selected as the backend model to do the inference. 


## Data

For demo purpose, we select LibriSpeech dataset (test-clean) to show workflow.


## Prerequisites

### Install Whisper

`pip3 install git+https://github.com/openai/whisper.git`
`git clone https://github.com/openai/whisper.git`

### install the required packages 

`pip3 install -r requirements`


## Usage

`python3 ./whisper_split_gt.py`

The script is able to split each file's ground truth and save to path.

`python3 ./whisper_infer.py`

The script inferences the inputs and generate transcribes. 

`python3 ./whisper_wer_cal.py`

The script is able to calculate output's WER and save results into csv file.

or all in one script

`./run.sh`

## Demo Results

There are several testing data put in [./LibriSpeech](./LibriSpeech) folder. 

### Input Audio & Ground Truth

#### input audio

[./LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac](./LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac)

#### ground truth

[./res/whisper_test_clean_gt_input_list/gt_txt/1089-134686-0000.txt](./res/whisper_test_clean_gt_input_list/gt_txt/1089-134686-0000.txt)

### Outputs Text

[./res/whisper_test_clean_opts/opt_txt/1089-134686-0000.txt](./res/whisper_test_clean_opts/opt_txt/1089-134686-0000.txt)

# Reference 

whisper official repo: [whisper](https://github.com/openai/whisper.git)


# Others

Please contact me, if you are interested in this project or have any questions.