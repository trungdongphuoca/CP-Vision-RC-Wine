import sys, os; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[1])); import config as cfg
import argparse
import os
import re
import torch
from transformers import TrainingArguments
from datasets import load_dataset

# Conditional import of unsloth
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

# Configuration
MAX_SEQ_LENGTH = 2048  # Max sequence length
DTYPE = None # None for auto detection
LOAD_IN_4BIT = True # Use 4bit quantization to reduce VRAM
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="Full training epochs. Default is 1.0 for real LoRA training.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override for quick smoke runs. Use 60 to reproduce the old demo run.")
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()

# 1. Load the model and tokenizer
def setup_model():
    if HAS_UNSLOTH:
        print("[INFO] Using Unsloth for training initialization.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        # 2. Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", 
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
    else:
        print("[INFO] Unsloth not available. Falling back to native Hugging Face PEFT + BitsAndBytes...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

        # Load in 4-bit using native bitsandbytes config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Prepare model for kbit training
        model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Standard dropout for native HF
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
    return model, tokenizer

def formatting_prompts_func(examples, tokenizer):
    instructions = examples["instruction"]
    thoughts = examples["thought"]
    responses = examples["response"]
    
    texts = []
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Master Sommelier. Analyze the user's request and determine the ideal structural profile of the wine. Then, output the Semantic ID of the perfect match, followed by a persuasive explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<thought>
{}
</thought>
{}<|eot_id|>"""
    
    for instruction, thought, response in zip(instructions, thoughts, responses):
        text = prompt_template.format(instruction, thought, response)
        texts.append(text)
        
    return { "text" : texts }

def main():
    args = parse_args()

    print("Setting up model and LoRA adapters...")
    model, tokenizer = setup_model()
    
    print("Loading datasets...")
    train_file = str(cfg.TRAIN_JSONL)
    val_file = str(cfg.VAL_JSONL)
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"Error: Dataset files not found. Run data_prep.py first.")
        return
        
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    
    # Map the formatting function over the datasets
    dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched = True)
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_train_dataset = dataset["train"].map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_val_dataset = dataset["validation"].map(tokenize_function, batched=True, remove_columns=dataset["validation"].column_names)
    
    # Select a small validation subset (e.g. 200 samples) to prevent slow evaluation steps
    tokenized_val_dataset = tokenized_val_dataset.select(range(min(200, len(tokenized_val_dataset))))
    
    from transformers import Trainer, DataCollatorForLanguageModeling

    print("Starting standard Trainer...")
    training_output_dir = cfg.RESULTS / "training_outputs"
    logging_dir = cfg.RESULTS / "training_logs"
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_ratio = 0.03,
        num_train_epochs = args.epochs,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = args.logging_steps,
        eval_strategy = "steps",
        eval_steps = args.eval_steps,
        save_strategy = "steps",
        save_steps = args.save_steps,
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = str(training_output_dir),
        logging_dir = str(logging_dir),
        report_to = "none",
    )

    trainer = Trainer(
        model = model,
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_val_dataset,
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args = training_args,
    )
    
    print("Training model...")
    trainer_stats = trainer.train()
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")
    if eval_loss is not None:
        print(f"Final eval_loss: {eval_loss:.4f}")
    
    print("Saving the LoRA adapters...")
    model.save_pretrained(str(cfg.LORA_MODEL))
    tokenizer.save_pretrained(str(cfg.LORA_MODEL))
    
    print(f"Fine-tuning complete. Adapters saved to {cfg.LORA_MODEL}.")

if __name__ == "__main__":
    main()
