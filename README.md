# DDSC
Detoxifying Large Language Models without Fine-Tuning: Leveraging Layer-Wise Constraints
## Overview

![DDSC](DDSC.pdf)

Detoxification in large language models (LLMs) remains a significant research challenge. Existing methods primarily rely on knowledge editing to diminish toxicity that has already been exposed by the model. These methods depend on high-quality toxic supervision data, are only applicable after toxicity has been exposed, and often lack generalization ability. This work innovatively proposes Detoxification with Dynamic Safety Constraints (DDSC), a novel method for reducing LLMs toxicity without parameter fine-tuning. Additionally, it introduces a toxicity prevention mode that enables the model to proactively prevent the output of toxic content before its toxicity is exposed. DDSC strengthens the token distribution of the safe layer while weakening that of hallucination and toxic layer during output generation. This effectively diminishes toxicity and enhances output safety. As a method that does not require parameter fine-tuning, DDSC offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing knowledge-editing detoxification methods for further performance improvement. Extensive experiments across representative open-source LLMs and public datasets demonstrate the effectiveness of DDSC, achieving state-of-the-art (SOTA) results when combined with existing knowledge-editing detoxification techniques.

## Setup

```

conda create -n DDSC python=3.9.7
conda activate DDSC
cd EasyEdit_DDSC
pip install -r requirements.txt

```

> ❗️❗️Replace the trasformers already installed with transformers in the DDSC package
Besides,if you intend to use Mistral, please update the `transformers` library to version 4.34.0 manually. You can use the following code: `pip install transformers==4.34.0`,and change the logits calculation part of greedy_search in transformers to the logits calculation part of greedy_search in DDSC.
Dataset download(Classification Task and Generation Task)：https://1drv.ms/f/c/0cb7116cbfb1bf89/EupbR5w5zsxBoXTHSU7PBwAB5dxIKtSpqT5GdThrto9ptw?e=0bcS2i

---
```

## Experiments

### Generation Task Arguments

| Argument          | Example           | Description   |
| ----------------- | ----------------- | ------------- |
| `--early-exit-layers`     | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32` | The candidate layers include safe_layers, toxic_layer, hallucination_layers and end_layer |
| `--edited_model`   | `llama2-7b-chat` | Vanilla LLM |
| `--hparams_dir`      | `LLaMA2-7B-Chat_PROMPT1.yaml ` | Parameter configuration.  |
| `--editing_method`    | `DINM` | Indicates the method for inserting DDSC into parameter updates |
| `--safety_classifier_dir`| `Roberta or GPT-4o` |A classifier that judges whether it is safe to generate sentences  |
| `--metrics_save_dir`   | `/LLAMA/test` | Where to store the output results. |

### Classification Task Arguments

| Argument          | Example           | Description   |
| ----------------- | ----------------- | ------------- |
| `--model-name`    | `huggyllama/llama-7b` | Specifies the model you want to use, currently we only support LLaMA-v1. |
| `--early-exit-layers`     | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32` | The candidate layers include safe_layers, toxic_layer, hallucination_layers and end_layer |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder. |
| `--output-path`   | `output-path.json` | Where to store the output results. |
| `--num-gpus`      | `1` | Number of GPUs to use, `1/2/4/8` for `7B/13B/30B/65B` model sizes respectively.  |

### Understanding `--early-exit-layers`

The `--early-exit-layers` argument takes a string containing a sequence of layer numbers separated by commas, with no spaces in between. By specifying different number of layers, we ues different modes.


| MODE  | Example (str)     | Description of DDSC Mode                                                                                     |
| ---------------------------| ------------- | ----------------------------------------------------------------------------------------------- |
| MODE-1                         | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32`    | **DDSC** with the last specified layer (i.e. `32`) as the `end_layer` and all the preceding layers (i.e. `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32`) as `toxic_layer`, `candidate_hallucination_layers` and `candidate_safe_layers`. |
| MODE-2                        | `0,2,28,31,32`      | **DDSC** with the last specified layer (i.e. `32`) as the `end_layer`, the layer having the greatest difference in hidden_states as the `toxic_layer` and all the preceding layers (i.e. `0,2,28,31`) as `candidate_hallucination_layers` and `candidate_safe_layers`. |layer output.       |


### SafeEdit (Generation Task)

#### LLaMA
```bash
python run_safety_editing_DDSC.py --editing_method DINM --early-exit-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 --edited_model llama2-7b-chat --hparams_dir /mnt/sdb/zjk/EasyEdit_DDSC/hparams/DINM/LLaMA2-7B-Chat_PROMPT1.yaml --safety_classifier_dir /mnt/sdb/zjk/EasyEdit_ori/SafeEdit-Safety-Classifier --metrics_save_dir /mnt/sdb/zjk/EasyEdit_DDSC/LLAMA/test

```

#### Mistral
```bash
python run_safety_editing_DDSC.py --editing_method DINM --early-exit-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 --edited_model mistral-7b --hparams_dir /mnt/sdb/zjk/EasyEdit_DDSC/hparams/DINM/Mistral-7b-V0.1_PROMPT1.yaml --safety_classifier_dir /mnt/sdb/zjk/EasyEdit_ori/SafeEdit-Safety-Classifier --metrics_save_dir /mnt/sdb/zjk/EasyEdit_DDSC/Mistral/test
```

### SafeEdit (Classification Task)

```bash
python classification.py --model-name huggyllama/llama-7b --early-exit-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32 --data-path /mnt/sdb/zjk/DoLa/factor/data/dola+dinm_all.json --output-path output.json --num-gpu 1

```

