from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_safety_edit_quality(
    ddsc,
    model,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    toxic_layer: int,
    generate_kwargs:typing.Dict,
    max_tokens = 1024, 
    max_output_tokens: int = 500,
    


) -> typing.Dict:

    batch = [record["prompt"]] + record['general_prompt']

    DS, DG_onlyQ, DG_otherA, DG_otherQ, DG_otherAQ = test_safety_gen(ddsc,model, tok, batch, device, max_tokens, max_output_tokens,toxic_layer, **generate_kwargs)
    ret = {
        "DS": DS,
        "DG_onlyQ": DG_onlyQ,
        "DG_otherA": DG_otherA,
        "DG_otherQ": DG_otherQ,
        "DG_otherAQ": DG_otherAQ
    }
    return ret

def ccks_compute_safety_edit_quality(
    model,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    max_tokens = 500,
    max_output_tokens: int = 400,
) -> typing.Dict:
    batch = [record["prompt"]] + record['general_prompt']
    DS, DG_otherAQ = test_safety_gen(model, tok, batch, device, max_tokens, max_output_tokens)
    ret = {
        "DS": DS,
        "DG_otherAQ": DG_otherAQ
    }
    return ret
