from meta_head import MetaLlamaForCausalLM, MetaheadModel
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger
from transformers import LlamaTokenizerFast, TrainingArguments
import json
from datasets import Dataset
from typing import Dict
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def get_model():
    pretrained_ckpt = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    model = MetaLlamaForCausalLM.from_pretrained(pretrained_ckpt, device_map={"": 0})
    logger.success("Load base model complete")

    # load the meta model
    model.meta_head = MetaheadModel.from_pretrained('saved_ckpt/meta-head').to(model.device)
    logger.success("Load meta model complete")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=64, lora_dropout=0.1
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.train_meta_head()
    lora_model.print_trainable_parameters()
    return lora_model

def get_tokenizer():
    pretrained_ckpt = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_ckpt)
    return tokenizer

def prompt_template(example: Dict) -> str:
    template = (
        "{prompt}\n\n### Answer: {output}"
    )
    text = template.format(prompt=example['prompt'], output=example['output'])
    return text
    
def get_data(subset: str):
    data = []
    with open(f'data/{subset}.jsonl') as fin:
        for line in fin:
            _data = json.loads(line)
            text = {"text": prompt_template(_data)}
            data.append(text)
    dataset = Dataset.from_list(data)
    return dataset

if __name__ == "__main__":
    model = get_model()

    # loading the tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    logger.success("Successfully load the tokenizer")

    # loading training data
    train_set = get_data('filtered_train')
    valid_set = get_data('valid')
    logger.success("Successfully load data")

    response_template_with_context = "\n### Answer:"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    logger.info("Initializing training configs")
    training_arguments = TrainingArguments(
        output_dir='testing_constant_v1',
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        bf16=False,
        max_steps=-1, # the number of training steps the model will take
        save_total_limit=2,
    )

    logger.info("Training ...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_set,
        eval_dataset=valid_set,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=data_collator,
    )

    trainer.train()

    logger.info("Saving ...")
    trainer.save_model('saved_ckpt/final_output_v7')
    model.save_meta_head('saved_ckpt/dimension-glue-v7')
