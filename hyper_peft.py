#!/usr/bin/env python3
from peft.tuners.lora import LoraLayer
from typing import Union, List
import torch.nn as nn
import importlib
import torch
import peft

assert peft.__version__ in ("0.2.0", "0.3.0", "0.3.0.dev0"), \
    "Definitly not works with 0.4.0 (Lora wastly changed), Not sure about earlier versions."


# HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models
# https://arxiv.org/abs/2307.06949
USE_HYPER = False
HYPER_A = 100
HYPER_B = 50


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


class HyperLinear(torch.nn.Module):
    def __init__(self, outer_size: int, hyper_size: int, r_size: int, is_A: bool):
        super().__init__()
        if is_A:  # outer >> hyper ← orthogonality in hyper
            self.layer = torch.nn.Linear(hyper_size, r_size, bias=False)
            hyper = torch.empty(size=(outer_size, hyper_size)).type_as(self.layer.weight)
            torch.nn.init.orthogonal_(hyper)
        else:  # outer << hyper ← orthogonality in outer
            self.layer = torch.nn.Linear(r_size, hyper_size, bias=False)
            hyper = torch.empty(size=(hyper_size, outer_size)).type_as(self.layer.weight)
            torch.nn.init.orthogonal_(hyper)
        self.register_buffer("hyper", hyper)
        self.is_A = is_A
        self.weight = self.layer.weight

    def forward(self, x: torch.Tensor):
        # print(x.dtype, self.hyper.dtype, self.layer.weight.dtype)
        if self.is_A:
            return self.layer(x @ self.hyper)
        return self.layer(x) @ self.hyper


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        use_hyper = USE_HYPER
        hyper_a = HYPER_A
        hyper_b = HYPER_B
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            if use_hyper:
                self.lora_A = HyperLinear(outer_size=in_features, hyper_size=hyper_a, r_size=r, is_A=True)
                self.lora_B = HyperLinear(outer_size=out_features, hyper_size=hyper_b, r_size=r, is_A=False)
            else:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T


if is_bnb_available:
    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
            # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            use_hyper = USE_HYPER
            hyper_a = HYPER_A
            hyper_b = HYPER_B
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                if use_hyper:
                    self.lora_A = HyperLinear(outer_size=in_features, hyper_size=hyper_a, r_size=r, is_A=True)
                    self.lora_B = HyperLinear(outer_size=out_features, hyper_size=hyper_b, r_size=r, is_A=False)
                else:
                    self.lora_A = nn.Linear(in_features, r, bias=False)
                    self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()


peft.tuners.lora.Linear.__init__ = Linear.__init__
if is_bnb_available:
    peft.tuners.lora.Linear8bitLt.__init__ = Linear8bitLt.__init__


def main():
    from transformers import LlamaConfig, LlamaForCausalLM
    load_in_8bit = True
    vocab_size = 4000
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=32
    )
    model = LlamaForCausalLM.from_pretrained(
        "del_model",
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # model = LlamaForCausalLM(config)
    # model.save_pretrained("del_model")
    # return

    if load_in_8bit:
        model = peft.prepare_model_for_int8_training(model)
    config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, config)
    # print(model)
    # for n, p in model.named_parameters():
    #     print(f"{n} | {p.requires_grad}")
    a = torch.randint(low=0, high=vocab_size, size=(1, 512), dtype=torch.long, device=next(model.parameters()).device)
    r = model(a, labels=a.clone())
    opt = torch.optim.Adam(params=model.parameters())
    r.loss.backward()
    opt.step()
    # model.save_pretrained("del_model")
    # print(model)


if __name__ == '__main__':
    main()
