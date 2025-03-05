# 簡化版TinyLlama訓練程式 - 不使用量化
# 安裝所需套件
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
        鞋子上的黑黴菌是一種常見的真菌，通常在潮濕環境中生長。它們以有機物質為食，會產生黑色或深綠色的孢子並釋放到空氣中。長期接觸可能導致過敏反應或皮膚問題，尤其對呼吸道敏感的人更應注意。黑黴菌生長時通常呈現黑點或黑斑，在皮革和帆布鞋上特別常見。"""
    },
    {
        "text": """### 指令：
        你是一位專門處理鞋子黴菌問題的專家。請回答以下問題：
        如何清除鞋子上的黴菌？

        ### 回應：
        清除鞋子上的黴菌需要幾個步驟：首先，在通風處用軟刷輕輕刷去表面黴菌，避免孢子擴散；接著，混合等量白醋和水，用軟布沾取溶液輕擦受感染區域；然後，使用75%酒精進行二次消毒；最後，徹底風乾至少24小時。對於皮革鞋，完成後應使用皮革保養油滋養皮質，防止乾裂。預防復發的方法包括：確保鞋子在存放前完全乾燥，使用矽膠乾燥劑，保持收納處通風。"""
    }
]

# 2. 主要訓練函數
def train_model():
    # 設置輸出目錄
    output_dir = "./simple_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型和tokenizer
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"正在載入模型：{model_id}")
    
    # 載入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型 - 使用半精度
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA配置
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 應用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 準備資料集
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # 分詞化資料
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 設定訓練參數
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
        fp16=True,
        report_to="tensorboard",
    )
    
    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                   'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                   'labels': torch.stack([f['input_ids'] for f in data])},
    )
    
    # 開始訓練
    print("開始訓練...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    return model, tokenizer

if __name__ == "__main__":
    print("開始簡化版TinyLlama訓練...")
    model, tokenizer = train_model()
    print("訓練完成!")