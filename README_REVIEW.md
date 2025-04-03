## Summary
We sincerely appreciate the reviewers' feedback and believe these clarifications will improve the paper. We will revise the manuscript accordingly.
## Reviewer 1


**A1:** Thank you for your valuable feedback. Regarding your statement that most of the benefits in the vanilla model using LLaMA2-7B-Chat-Uncensored indeed come from SFT, I acknowledge that applying DSCD alone may not achieve state-of-the-art (SOTA) performance. However, the strength of my method lies in the advantages highlighted in my abstract—lightweight, high compatibility, and plug-and-play capabilities. DSCD can be seamlessly integrated with existing SOTA detoxification methods to further enhance model safety.

**A2:** Thank you for your valuable feedback. Regarding the observation that the SFT variant takes significantly less time than the vanilla model, this raised some doubts in my experiments as well. After conducting multiple verifications and ensuring no errors in the experimental setup, the result remained unchanged. I then directly examined the generated responses and found that the number of tokens generated was noticeably fewer than in earlier methods (DINM, DPO, and SafeDecoding). Upon further reflection, I realized that the SFT method itself tends to generate shorter responses because it uses supervised learning (CE Loss), which makes the model more sensitive to the EOS token (end-of-sequence), leading it to terminate generation earlier. Additionally, since the vanilla model is a chat model (as shown in Table 4 of my paper, where the time for chat models is lower than that of other vanilla models), the fine-tuning of the chat model encourages simpler, conversational responses. Therefore, after ensuring the experimental setup was correct and considering the principles of SFT and the characteristics of the vanilla model, I believe this result is reasonable.

Regarding the DPO method, the dataset labels longer responses as optimal, which leads to the generation of longer answers after training. The DINM method, on the other hand, involves parameter tuning for each question’s safe and unsafe generations, resulting in a higher time cost. DSCD_MODE2, due to its lack of parameter tuning and the fact that its final output is derived from contrastive decoding, has a speed comparable to the vanilla model. The final results are consistent with my analysis. I can also provide you with the SFT and DPO parameters used in my training to help you reproduce the results from my experiments.

**llama2-7b-chat-uncensored_lora_sft:**
### model
model_name_or_path: /mnt/sdb/zjk/ALL-Models/llama2-7b-uncensored-chat
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: SafeEdit
template: llama2
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/llama2-7b-uncensored-chat/lora/sft_safeedit
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: alpaca_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500

**llama2-7b-chat-uncensored_lora_dpo.yaml:**
### model
model_name_or_path: /mnt/sdb/zjk/ALL-Models/llama2-7b-chat-uncensored
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: SafeEdit_dpo
template: llama2
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/llama2-7b-chat-uncensored/lora/dpo
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# eval_dataset: dpo_en_demo
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
