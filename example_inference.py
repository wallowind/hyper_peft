#!/usr/bin/env python3
import fire
import torch

from transformers import LlamaForCausalLM, LlamaTokenizer


import peft
import hyper_peft
# --- IMPORTANT ---
# This`hyper parameters` does't save via peft's `save_pretrained`
# Therefore You must ensure those are the same as in training
hyper_peft.USE_HYPER = True
hyper_peft.HYPER_A = 100
hyper_peft.HYPER_B = 50

peft.tuners.lora.Linear.__init__ = hyper_peft.Linear.__init__
peft.tuners.lora.Linear8bitLt.__init__ = hyper_peft.Linear8bitLt.__init__


def evaluate(
    base_model: str = "/data/models/vicuna-7b",
    lora_model: str = "/data/models/trained_vicuna_hyper",
    load_in_8bit: bool = False,
):
    # Model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Trainer save adapter and config at end of thraining in parent folder
    # But You can also use any intermediate checkpoints directly
    lora_dir = f"{lora_model}/checkpoint-3"
    model = peft.PeftModel.from_pretrained(model, lora_dir)
    # NOTE: Peft's weights are always in float32, and you will get a dtype error
    # for all models that have float16 weights (Llama and Vicuna included).
    # The types are only correctly cast for the 8-bit Peft's layer.
    # A simple (and probably not so good) way to fix this is to explicitly convert wheights.
    # for m in model.modules():
    #     if isinstance(m, hyper_peft.HyperLinear):
    #         m.layer.weight.data = m.layer.weight.data.to(torch.float16)
    #         m.hyper = m.hyper.to(torch.float16)
    # print(model)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    device = next(model.parameters()).device
    while True:
        prompt = input("User: ")
        if len(prompt) == 0:
            break
        prompt = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        # https://huggingface.co/docs/transformers/v4.30.0/main_classes/text_generation
        result = model.generate(
            inputs=prompt,
            max_new_tokens=64,
            num_beams=3,
            do_sample=False,
            eos_token_id=2,
        )
        print(f"Model: {tokenizer.decode(result.view(-1).tolist())}")
    return


if __name__ == '__main__':
    fire.Fire(evaluate)
