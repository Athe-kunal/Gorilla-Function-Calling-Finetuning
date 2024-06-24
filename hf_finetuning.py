import json
from datasets import load_dataset,Dataset
from tqdm import tqdm
import yaml
import ast
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
from dotenv import load_dotenv,find_dotenv
import os
import argparse

parser = argparse.ArgumentParser(
                    prog='HF-Finetuning',
                    description='Finetuning function calling models with HF')

parser.add_argument("-dn","--dataset_name",help="Name of the dataset",default="gorilla")
parser.add_argument("-nte","--num_train_epochs",help="Number of training epochs",default=3)
parser.add_argument("-od","--output_dir",help="Output directory name")
parser.add_argument("-rn","--run_name",help="Wandb run name")
parser.add_argument("-msl","--max_seq_length",help="Maximum sequence length",default=3076)
args = parser.parse_args()



_ = load_dotenv(find_dotenv(),override=True)

wandb.login(key=os.environ['WANDB_API_KEY'])
system_message = """You are an text to python function translator. Users will ask you questions in English and you will generate a python function based on the provided FUNCTIONS.
FUNCTIONS:
{functions}"""
if args.dataset_name == "gorilla":
    train_data = []
    with open('gorilla_openfunctions_v1_train.json', 'r') as file:
        for line in file:
            train_data.append(json.loads(line.strip()))
    # test_data = []
    with open('gorilla_openfunctions_v1_test.json', 'r') as file:
        test_data = json.load(file)
    def create_conversation(sample):
        return {
        "messages": [
        {"role": "system", "content": system_message.format(functions=sample["Functions"])},
        {"role": "user", "content": sample["Instruction"]},
        {"role": "assistant", "content": sample["Output"]}
        ]
    }
    def json_to_yaml(json_data):
        for data in tqdm(json_data):
            curr_func_yaml = ""
            for func in data['Functions']:
                curr_func_yaml+=yaml.dump(ast.literal_eval(func)) + "\n\n"
            data.update({"yaml_function":curr_func_yaml})
        return json_data

    for td in train_data:
        td['Output'] = td['Output'][0]

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_data))
    train_dataset = train_dataset.map(create_conversation, remove_columns=train_dataset.features,batched=False)
# def process_train_data(train_data):
#     for td in train_data:
#         output = td['Output']
#         td['Functions'] = str(td['Functions'])
#         if isinstance(output,str):
#             pass
#         elif isinstance(output,list):
#             output = output[0]
#         td['Output'] = output
#     return train_data

# train_data = process_train_data(train_data)

# llama_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are an text to python function translator. Users will ask you questions in English and you will generate a python function based on the provided FUNCTIONS.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>

# ### FUNCTIONS: {functions}<|eot_id|>

# <|start_header_id|>user<|end_header_id|>

# ### Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# {output}<|eot_id|>
# """


# train_dataset.to_json("train_dataset.json", orient="records")

# Hugging Face model id
model_id = "mistralai/Mistral-7B-v0.1" # or `mistralai/Mistral-7B-v0.1`
 
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

model, tokenizer = setup_chat_format(model, tokenizer)

from peft import LoraConfig
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=args.output_dir, # directory to save and repository id
    num_train_epochs=args.num_train_epochs,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    per_device_eval_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=False,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="wandb",                
    run_name=args.run_name
)


max_seq_length = args.max_seq_length # max sequence length for model and packing of the dataset
 
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()
 
# save model
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()