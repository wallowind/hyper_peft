#!/usr/bin/env python3
import os
import fire
import torch

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

from transformers import LlamaForCausalLM, Trainer, TrainingArguments


import peft
import hyper_peft

hyper_peft.USE_HYPER = True
hyper_peft.HYPER_A = 100
hyper_peft.HYPER_B = 50

peft.tuners.lora.Linear.__init__ = hyper_peft.Linear.__init__
peft.tuners.lora.Linear8bitLt.__init__ = hyper_peft.Linear8bitLt.__init__


# Save only peft's adapter and config in Trainer
old_save = Trainer._save
def hyper_save(self, output_dir=None, state_dict=None):
    # Probably won't work in most cases, but does work in this example
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(self.model, peft.PeftModel):
        self.model.save_pretrained(output_dir)
    else:
        old_save(self, output_dir, state_dict)
Trainer._save = hyper_save


def train(
    base_model: str = "/data/models/vicuna-7b",
    data_path: str = "data/booksdata/data_lama.pt",
    output_dir: str = "/data/models/trained_vicuna_hyper",
    # training hyperparams
    load_in_8bit: bool = True,
    batch_size: int = 16,
    micro_batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,  # 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
):
    # Data
    def retokenize(v: torch.Tensor, pad_to_size: int = 0):
        result = dict()
        if pad_to_size:
            assert pad_to_size >= v.size(0), "Truncating!"
            diff = pad_to_size - v.size(0)
            f = torch.full(size=(pad_to_size,), fill_value=0, dtype=v.dtype)
            f[diff:] += v
        else:
            f = v
            diff = 0
        a = f
        b = a.clone()
        b[:diff] = -100
        c = torch.ones_like(a)
        c[:diff] = 0
        result["input_ids"] = a
        result["labels"] = b
        result["attention_mask"] = c
        return result

    class DataList(torch.utils.data.Dataset):
        """ dataset for list """

        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            # print(idx)
            if idx < len(self.data_list):
                return self.data_list[idx]
            return None

    data = torch.load("data_lama.pt")  # already tokenized and prepared (not ready for batching though)
    pad_to_size = sorted(data, key=lambda v: v.size())[-1].size(0)
    data = [retokenize(v, pad_to_size) for v in data]
    # print(f"Data len: {len(data)}")
    train_data = DataList(data)

    # Model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if load_in_8bit:  # Also turn on gradient_checkpointing
        model = prepare_model_for_int8_training(model)

    # Hack to replace model's state_dict to peft's only state_dict
    # Not needed in this example
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # What is this?
    # model.config.use_cache = False

    # Lora
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    # print(model)

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    # Only for torch >2.0 (but does't work anyway...)
    # model = torch.compile(model)

    # Trainer
    gradient_accumulation_steps = batch_size // micro_batch_size
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=500 if val_set_size > 0 else None,
            save_steps=3,  # 302,
            output_dir=output_dir,
            save_total_limit=40,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            report_to="none",
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
    )

    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
