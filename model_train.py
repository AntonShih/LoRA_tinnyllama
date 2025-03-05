# 簡化版 TinyLlama 訓練程式 - CPU 版
# pip install transformers datasets peft accelerate tensorboard

import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# 1. 準備簡單資料集
data = [
    {
        "text": """### 指令：
        你是一位專門處理鞋子黴菌問題的專家。請回答以下問題：
        什麼是鞋子上的黑黴菌？

        ### 回應：
        鞋子上的黑黴菌是一種常見的真菌，通常在潮濕環境中生長。它們以有機物質為食，會產生黑色或深綠色的孢子並釋放到空氣中。"""
    },
    {
        "text": """### 指令：
        你是一位專門處理鞋子黴菌問題的專家。請回答以下問題：
        如何清除鞋子上的黴菌？

        ### 回應：
        清除鞋子上的黴菌需要幾個步驟：首先，在通風處用軟刷輕輕刷去表面黴菌，避免孢子擴散；接著，混合等量白醋和水，用軟布沾取溶液輕擦受感染區域；最後，徹底風乾至少24小時。"""
    }
]

# 2. 主要訓練函數
def train_model():
    output_dir = "./simple_results"
    os.makedirs(output_dir, exist_ok=True)
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"正在載入模型：{model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, force_download=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 設定 torch dtype（優先使用 bf16）
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    # 載入模型（強制使用 CPU）
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": "cpu"}  # 明確指定使用 CPU
    )
    
    # LoRA 設定
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 避免不支援模組報錯
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,  # CPU 訓練使用 bf16
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.tensor([f['input_ids'] for f in data]),
            'attention_mask': torch.tensor([f['attention_mask'] for f in data]),
            'labels': torch.tensor([f['input_ids'] for f in data]),
        },
    )
    
    print("開始訓練...")
    trainer.train()
    
    print("保存模型...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    return model, tokenizer

if __name__ == "__main__":
    print("開始簡化版 TinyLlama 訓練（CPU 版）...")
    model, tokenizer = train_model()
    print("訓練完成!")

