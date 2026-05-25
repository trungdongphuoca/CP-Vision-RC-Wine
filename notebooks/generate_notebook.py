import json

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Huấn luyện LLM Recommend System (Unsloth + Llama-3)\n",
        "Notebook này được tối ưu để chạy trên Google Colab với GPU (T4/A100). Hãy đảm bảo bạn đã bật GPU: **Runtime -> Change runtime type -> T4 GPU**."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 1. Cài đặt các thư viện cần thiết"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "%%capture\n",
        "!pip install unsloth\n",
        "!pip install --no-deps xformers \"trl<0.9.0\" peft accelerate bitsandbytes\n",
        "!pip install datasets"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 2. Tải file dataset lên Colab\n",
        "Bạn cần chạy file `data_prep.py` ở máy tính cá nhân để tạo ra file `wine_training_dataset.jsonl`. Sau đó chạy ô dưới đây để upload file đó lên Colab."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "\n",
        "print('Upload file wine_train_130k.jsonl:')\n",
        "uploaded = files.upload()\n",
        "print('Upload file wine_val_130k.jsonl:')\n",
        "uploaded = files.upload()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 3. Load Model (Llama-3-8B 4-bit) & Thêm LoRA Adapters"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from unsloth import FastLanguageModel\n",
        "import torch\n",
        "\n",
        "max_seq_length = 2048 \n",
        "dtype = None \n",
        "load_in_4bit = True \n",
        "\n",
        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
        "    model_name = \"unsloth/llama-3-8b-bnb-4bit\",\n",
        "    max_seq_length = max_seq_length,\n",
        "    dtype = dtype,\n",
        "    load_in_4bit = load_in_4bit,\n",
        ")\n",
        "\n",
        "model = FastLanguageModel.get_peft_model(\n",
        "    model,\n",
        "    r = 16,\n",
        "    target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
        "                      \"gate_proj\", \"up_proj\", \"down_proj\",],\n",
        "    lora_alpha = 16,\n",
        "    lora_dropout = 0, \n",
        "    bias = \"none\",    \n",
        "    use_gradient_checkpointing = \"unsloth\",\n",
        "    random_state = 3407,\n",
        "    use_rslora = False,\n",
        "    loftq_config = None,\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 4. Chuẩn bị Dataset theo định dạng"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from datasets import load_dataset\n",
        "\n",
        "EOS_TOKEN = tokenizer.eos_token\n",
        "\n",
        "prompt_template = \"\"\"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n",
        "You are a Master Sommelier. Analyze the user's request and determine the ideal structural profile of the wine. Then, output the Semantic ID of the perfect match, followed by a persuasive explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>\n",
        "{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "<thought>\n",
        "{}\n",
        "</thought>\n",
        "{}<|eot_id|>\"\"\"\n",
        "\n",
        "dataset = load_dataset(\"json\", data_files={\"train\": \"wine_train_130k.jsonl\", \"validation\": \"wine_val_130k.jsonl\"})\n",
        "\n",
        "def formatting_prompts_func(examples):\n",
        "    instructions = examples[\"instruction\"]\n",
        "    thoughts = examples[\"thought\"]\n",
        "    responses = examples[\"response\"]\n",
        "    texts = []\n",
        "    for instruction, thought, response in zip(instructions, thoughts, responses):\n",
        "        text = prompt_template.format(instruction, thought, response) + EOS_TOKEN\n",
        "        texts.append(text)\n",
        "    return { \"text\" : texts, }\n",
        "\n",
        "dataset = dataset.map(formatting_prompts_func, batched = True)\n",
        "\n",
        "def tokenize_function(examples):\n",
        "    return tokenizer(examples[\"text\"], truncation=True, max_length=max_seq_length, padding=\"max_length\")\n",
        "\n",
        "tokenized_train_dataset = dataset[\"train\"].map(tokenize_function, batched=True, remove_columns=dataset[\"train\"].column_names)\n",
        "tokenized_val_dataset = dataset[\"validation\"].map(tokenize_function, batched=True, remove_columns=dataset[\"validation\"].column_names)\n",
        "\n",
        "print(\"Training samples:\", len(tokenized_train_dataset))\n",
        "print(\"Validation samples:\", len(tokenized_val_dataset))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 5. Tiến hành Training"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling\n",
        "from unsloth import is_bfloat16_supported\n",
        "\n",
        "trainer = Trainer(\n",
        "    model = model,\n",
        "    train_dataset = tokenized_train_dataset,\n",
        "    eval_dataset = tokenized_val_dataset,\n",
        "    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),\n",
        "    args = TrainingArguments(\n",
        "        per_device_train_batch_size = 2,\n",
        "        per_device_eval_batch_size = 2,\n",
        "        gradient_accumulation_steps = 4,\n",
        "        warmup_steps = 5,\n",
        "        max_steps = 100,\n",
        "        learning_rate = 2e-4,\n",
        "        fp16 = not is_bfloat16_supported(),\n",
        "        bf16 = is_bfloat16_supported(),\n",
        "        logging_steps = 10,\n",
        "        evaluation_strategy = \"steps\",\n",
        "        eval_steps = 20,\n",
        "        optim = \"adamw_8bit\",\n",
        "        weight_decay = 0.01,\n",
        "        lr_scheduler_type = \"linear\",\n",
        "        seed = 3407,\n",
        "        output_dir = \"outputs\",\n",
        "    ),\n",
        ")\n",
        "\n",
        "trainer_stats = trainer.train()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "### 6. Lưu và tải mô hình về máy"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "model.save_pretrained(\"lora_wine_model\")\n",
        "tokenizer.save_pretrained(\"lora_wine_model\")\n",
        "\n",
        "# Nén folder lại\n",
        "!zip -r lora_wine_model.zip lora_wine_model\n",
        "\n",
        "# Tải xuống máy cá nhân\n",
        "from google.colab import files\n",
        "files.download('lora_wine_model.zip')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("Wine_Finetune_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)
print("Notebook created successfully.")
