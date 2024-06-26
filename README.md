# Enhancing Visual Question Answering through Question-Driven Image Captions as Prompts
Work  in Progress.

## Introduction
This is the official code for the paper "Enhancing Visual Question Answering through Question-Driven Image Captions as Prompts". The code evaluates the effect of using image captions with LLMs for zero-shot Visual Question Answering (VQA).

### Directory Structure

- **/predictions**: This directory contains the predictions generated by GPT-3.5 QA, CogVLM and BLIP-2 VQA models.
- **/captions**: This directory contains the image caption generated by several approaches.
```
cd captions-in-VQA
pip install -r requirements.txt
```

### Create image captions
The following script creates question-driven captions using KeyBERT(https://github.com/MaartenGr/KeyBERT) and CogVLM (https://github.com/THUDM/CogVLM).
You should install the requirements for these two packages before running the script.
The script is a modified version of https://github.com/THUDM/CogVLM/blob/main/basic_demo/cli_demo_hf.py 
```
python write_qd_captions.py --from_pretrained THUDM/cogvlm-chat-hf --bf16 --quant 4 --image-path path-to-images --question-path path-to-questions
```

### Generate answers through GPT-3.5 using image captions
Modify the OPENAI_API_KEY in the script before running.
```
# generate answers using question-driven captions
python qa.py --caption-file captions/cogvlm_chat_qd_descriptions.json --question-file path-to-questions
```

### Evaluate VQA predictions
```
# evaluate question-driven image captioning in VQA
python evaluation.py --prediction-file predictions/cogvlmqd_cap_gpt35.json --question-file path-to-questions
```

## Zero-shot Evaluation Results
We evaluate several VQA approaches on GQA testdev-balanced accross various question types.

| Question types | CogVLM-C Cap. + GPT-3.5 | CogVLM-V Cap. + GPT-3.5 | CogVLM QD Cap. + GPT-3.5 | CogVLM SB Cap. + GPT-3.5 | FuseCap + GPT-3.5 | BLIP-2 Cap. + GPT-3.5 |
|----------------|-------------------------|-------------------------|--------------------------|--------------------------|-------------------|-----------------------|
| verify         | 63.01                   | 58.53                   | 66.83                    | 61.06                    | 53.60             | 55.82                 |
| query          | 36.91                   | 31.08                   | 38.34                    | 31.51                    | 29.61             | 31.87                 |
| choose         | 65.25                   | 60.90                   | 65.51                    | 60.90                    | 58.07             | 60.82                 |
| logical        | 59.51                   | 60.29                   | 59.07                    | 58.46                    | 57.07             | 56.07                 |
| compare        | 51.78                   | 51.95                   | 51.95                    | 49.07                    | 54.50             | 48.22                 |
| object         | 61.95                   | 63.24                   | 59.13                    | 58.87                    | 59.38             | 58.35                 |
| attribute      | 51.75                   | 46.42                   | 54.62                    | 50.80                    | 45.11             | 46.63                 |
| category       | 47.35                   | 44.21                   | 50.39                    | 42.56                    | 43.52             | 42.47                 |
| relation       | 42.56                   | 38.32                   | 42.97                    | 35.76                    | 34.98             | 37.23                 |
| global         | 49.04                   | 45.86                   | 45.86                    | 44.59                    | 43.95             | 45.22                 | 
| TOTAL          | 48.06                   | 43.83                   | 49.50                    | 44.12                    | 41.58             | 42.99                 | 




