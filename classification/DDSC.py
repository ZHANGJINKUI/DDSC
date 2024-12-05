import argparse
import time
import csv
import tqdm
import os
import json
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np

class DDSC:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token_id = tok.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True,output_hidden_states=True, device_map='auto', **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tok

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, end_layer=None, hallucination_layer=None, candidate_hallucination_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():

            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, ddsc_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'ddsc-static':
                assert end_layer is not None, "end_layer must be specified"
                assert hallucination_layer is not None, "hallucination_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, ddsc_decoding=True,
                                    end_layer=end_layer, hallucination_layer=hallucination_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == 'ddsc':
                assert end_layer is not None, "end_layer must be specified"
                assert candidate_hallucination_layers is not None, "candidate_hallucination_layers must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, ddsc_decoding=True,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                                        end_layer=end_layer, hallucination_layer=None, candidate_hallucination_layers=candidate_hallucination_layers, **kwargs,)
                hallucination_layer_dist = outputs.hallucination_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (hallucination_layer_dist if mode == 'ddsc' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

#Only hallucination layer
    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, end_layer=None, hallucination_layer=None, candidate_hallucination_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'ddsc-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[hallucination_layer, end_layer],
                )

                assert hallucination_layer is not None
                hallucination_logits = dict_outputs[hallucination_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                hallucination_logits = hallucination_logits.log_softmax(dim=-1)
                diff_logits = final_logits - hallucination_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'ddsc':
                hallucination_layer_dist = {l:0 for l in candidate_hallucination_layers}
                picked_logits = []
                result_dict = {}
                hallucination_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_hallucination_layers+[end_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all hallucination_layers into a new dimension
                    stacked_hallucination_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_hallucination_layers], dim=0)

                    # 2. Calculate the softmax values for end_layer and all hallucination_layers
                    # print("end_layer:"+str(end_layer))
                    softmax_end_layer = F.softmax(dict_outputs[end_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_hallucination_layers = F.softmax(stacked_hallucination_layers, dim=-1)  # shape: (num_hallucination_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_end_layer[None, :, :] + softmax_hallucination_layers)  # shape: (num_hallucination_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_end_layer = F.log_softmax(dict_outputs[end_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_hallucination_layers = F.log_softmax(stacked_hallucination_layers, dim=-1)  # shape: (num_hallucination_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_end_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_hallucination_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_hallucination_layers, M, reduction='none').mean(-1)  # shape: (num_hallucination_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_hallucination_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_hallucination_layers,)
                    hallucination_layer = candidate_hallucination_layers[int(js_divs.argmax().cpu().item())]
                    hallucination_layer_dist[hallucination_layer] += 1

                    hallucination_layers.append(hallucination_layer)

                hallucination_logits = torch.zeros_like(dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(hallucination_layers):
                   hallucination_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                hallucination_logits = hallucination_logits.log_softmax(dim=-1)
                diff_logits = final_logits - hallucination_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                mask = diff_logits != -1000

                indices = torch.nonzero(mask, as_tuple=True)

                values = diff_logits[mask]


                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (hallucination_layer_dist if mode == 'ddsc' else None)
    
#Toxic regions

    def Logits_Distribution(self, input_text1, input_text2,toxic_layer, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, end_layer=None, hallucination_layer=None, candidate_hallucination_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):

        with torch.no_grad():

            input_text = input_text1 + input_text2


            input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(self.device)#Prefix and the security answers you want to generate
                
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt", padding=True).input_ids.to(self.device)#Prefix

            continue_ids = input_ids[0, prefix_ids.shape[-1]:]

            safe_layers_dist =candidate_hallucination_layers+[end_layer]
            safe_layer_dist = {l:0 for l in safe_layers_dist}
            hallucination_layer_dist = {l:0 for l in candidate_hallucination_layers}


            picked_logits = []
            result_dict = {}
            hallucination_layers = []
            safe_layers = []
            
            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=candidate_hallucination_layers+[end_layer],
                
            )

            for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):

                # Pick the less like layer to contrast with
                # 1. Stacking all safe_layers into a new dimension

                stacked_hallucination_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_hallucination_layers], dim=0)

                stacked_safe_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in safe_layers_dist], dim=0)
                # 2. Calculate the softmax values for toxic_layer,end_layer, hallucination_layers and safe_layers

                softmax_toxic_layer = F.softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                softmax_end_layer = F.softmax(dict_outputs[end_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                
                softmax_hallucination_layers = F.softmax(stacked_hallucination_layers, dim=-1)  # shape: (num_safe_layers, batch_size, num_features)

                softmax_safe_layers = F.softmax(stacked_safe_layers, dim=-1)  # shape: (num_safe_layers, batch_size, num_features)

                # 3. Calculate M,N the average distribution
                M = 0.5 * (softmax_toxic_layer[None, :, :]+ softmax_safe_layers)  # shape: (num_safe_layers, batch_size, num_features)
                N = 0.5 * (softmax_end_layer[None, :, :] + softmax_hallucination_layers)  # shape: (num_hallucination_layers, batch_size, num_features)


                # 4. Calculate log-softmax for the KL divergence
                log_softmax_toxic_layer = F.log_softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                log_softmax_safe_layers = F.log_softmax(stacked_safe_layers, dim=-1)  # shape: (num_safe_layers, batch_size, num_features)

                log_softmax_end_layer = F.log_softmax(dict_outputs[end_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                
                log_softmax_hallucination_layers = F.log_softmax(stacked_hallucination_layers, dim=-1)  # shape: (num_hallucination_layers, batch_size, num_features)
                # 5. Calculate the KL divergences and then the JS divergences
                kl3 = F.kl_div(log_softmax_end_layer[None, :, :], N, reduction='none').mean(-1)  # shape: (num_hallucination_layers, batch_size)
                kl4 = F.kl_div(log_softmax_hallucination_layers, N, reduction='none').mean(-1)  # shape: (num_hallucination_layers, batch_size)
                js_divs_E_H = 0.5 * (kl3 + kl4)  # shape: (num_hallucination_layers, batch_size)

                
                kl1 = F.kl_div(log_softmax_toxic_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_safe_layers, batch_size)
                kl2 = F.kl_div(log_softmax_safe_layers, M, reduction='none').mean(-1)  # shape: (num_safe_layers, batch_size)
                js_divs_S_T = 0.5 * (kl1 + kl2)  # shape: (num_safe_layers, batch_size)

                # 6. Reduce the batchmean
                js_divs_safe_mean = js_divs_S_T.mean(-1)  # shape: (num_safe_layers,)
                js_divs_safe_mean[0] = float('-inf')
                top_two_indices = js_divs_safe_mean.topk(2).indices.cpu().tolist()

                # Select safe_layer and ensure that layer 0 is not selected
                safe_layer = safe_layers_dist[top_two_indices[0]]
                
                safe_layer_dist[safe_layer] += 1

                js_divs_E_H = js_divs_E_H.mean(-1)  # shape: (num_hallucination_layers,)
                
                hallucination_layer = candidate_hallucination_layers[int(js_divs_E_H.argmax().cpu().item())]
                
                hallucination_layer_dist[hallucination_layer] += 1
                
                hallucination_layers.append(hallucination_layer)
                
                safe_layers.append(safe_layer)

            safe_logits = torch.zeros_like(dict_outputs[toxic_layer][0, prefix_ids.shape[-1] - 1:-1])

            hallucination_logits = torch.zeros_like(dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1:-1])
            
            bad_logits = torch.zeros_like(dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1:-1])
            for i, (safe_layer, hallucination_layer) in enumerate(zip(safe_layers, hallucination_layers)):
                hallucination_logits[i] = dict_outputs[hallucination_layer][0, prefix_ids.shape[-1] - 1 + i]
                safe_logits[i] = dict_outputs[safe_layer][0, prefix_ids.shape[-1] - 1 + i]
                bad_logits[i] =  hallucination_logits[i] - safe_logits[i]
                
            toxic_logits = dict_outputs[toxic_layer][0,prefix_ids.shape[-1] - 1:-1]
            final_logits = dict_outputs[end_layer][0, prefix_ids.shape[-1] - 1:-1]
            final_logits = final_logits.log_softmax(dim=-1)       
            hallucination_logits = hallucination_logits.log_softmax(dim=-1)        
            bad_logits = bad_logits+toxic_logits#Toxic regions include toxic_layer, safe_layers and hallucination_layers
            bad_logits = bad_logits.log_softmax(dim=-1)
            safe_logits = safe_logits.log_softmax(dim=-1)

            diff_logits = final_logits-hallucination_logits

            if post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)
                
            if relative_top > 0.0:

                relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)#

                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                true_indices = torch.nonzero(relative_top_mask, as_tuple=False)

                total_elements = final_logits.numel()
                selected_elements = true_indices.size(0)

                print("Percentage of selected elements:", selected_elements / total_elements * 100, "%")


            log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            diff_logits = diff_logits.unsqueeze(0)


        return log_probs,(hallucination_layer_dist if mode == 'ddsc' else None)            
