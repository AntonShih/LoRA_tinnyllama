import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from peft import (
    LoraConfig, 
    get_peft_model
)

from datasets import Dataset

# 直接在這裡定義資料
basic_training_data = [
    {
        "category": "黴菌基礎知識",
        "question": "鞋子為什麼會發霉？",
        "answer": "鞋子發霉主要是因為黴菌在適宜的條件下生長。這些條件包括濕度超過60%、溫暖的環境（20-30°C）以及有機物質（如皮革、布料）作為營養來源。鞋子經常接觸汗水和外部濕氣，且通常存放在通風不良的環境中，這些因素共同創造了黴菌生長的理想條件。"
    },
    {
        "category": "黴菌基礎知識",
        "question": "鞋子上的黴菌是從哪裡來的？",
        "answer": "鞋子上的黴菌主要來自空氣中懸浮的黴菌孢子，這些孢子無處不在。當孢子落在濕潤的鞋面或鞋內，遇到適合的溫度和營養源時，就會開始生長繁殖。此外，鞋子接觸地面時也可能從土壤中帶入黴菌孢子，或是從已感染黴菌的環境（如潮濕的浴室地板）轉移到鞋子上。"
    },
    # 添加更多訓練數據...
]

# 轉換資料和標記函數需要修改
def convert_to_instruction_format(data):
    """將原始資料轉換為模型訓練所需的格式"""
    print("轉換前資料型態:", type(data))
    print("轉換前資料是否為列表:", isinstance(data, list))
    print("轉換前第一筆資料:", data[0] if len(data) > 0 else "無資料")
    
    instruction_data = []
    
    for i, item in enumerate(data):
        print(f"處理第 {i+1} 筆資料, 型態: {type(item)}")
        
        instruction_item = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"],
            "category": item["category"]
        }
        instruction_data.append(instruction_item)
    
    print("轉換後資料大小:", len(instruction_data))
    print("轉換後第一筆資料:", instruction_data[0] if len(instruction_data) > 0 else "無資料")
    
    if i == 1:
            print("\n第二筆資料轉換結果:")
            print("原始資料:", item)
            print("轉換後資料:", instruction_item)
            print("轉換後資料類型:", type(instruction_item))
            print("各欄位類型:")
            print("  - instruction:", type(instruction_item["instruction"]))
            print("  - input:", type(instruction_item["input"]))
            print("  - output:", type(instruction_item["output"]))
            print("  - category:", type(instruction_item["category"]))
            print()
    
    return instruction_data

def tokenize_function(examples, tokenizer, max_length=512):
    """將文本轉換為模型能夠處理的標記形式"""
    # 構建提示文本
    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        output = examples["output"][i]
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            prompt = f"Instruction: {instruction}\nOutput: {output}"
        
        prompts.append(prompt)
    
    # 標記化
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 添加標籤，用於自回歸訓練
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# 主要程序
def main():
    print("開始準備訓練資料...")
    
    # 轉換資料
    instruction_data = convert_to_instruction_format(basic_training_data)
    print(f"轉換後資料數量: {len(instruction_data)}")
    print(f"資料範例: {instruction_data[0]}")
    
    # 轉換成 Hugging Face Dataset 格式
    dataset = Dataset.from_list(instruction_data)
    print(f"Dataset 大小: {len(dataset)}")
    
    # 載入模型和 tokenizer
    model_path = "./data/model/tinyllama-local"
    print(f"正在載入模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 對資料集進行標記化
    print("正在標記化資料集...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names  # 移除原始文本列
    )
    print(f"標記化後的資料集大小: {len(tokenized_dataset)}")
    print(f"標記化後的資料集欄位: {tokenized_dataset.column_names}")
    
    # 載入模型
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": "cpu"}
    )
    
    # 設定 LoRA
    print("設定 LoRA 配置...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 設定訓練參數
    output_dir = "./results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=1,  # 降低批次大小
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1,
        save_steps=10,
        warmup_ratio=0.03,
        save_total_limit=2,
        report_to="none"  # 減少額外的報告
    )
    
    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # 開始訓練
    print("開始訓練...")
    trainer.train()
    
    # 保存模型
    lora_output_dir = os.path.join(output_dir, "lora-weights")
    os.makedirs(lora_output_dir, exist_ok=True)
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    
    print(f"訓練完成，模型已保存至: {lora_output_dir}")
    
    # 測試模型
    test_prompts = [
        "如何防止鞋子發霉？",
        "鞋子上的黴菌有什麼危害？"
    ]
    
    print("\n開始測試模型...")
    for prompt in test_prompts:
        print(f"\n問題: {prompt}")
        
        inputs = tokenizer(f"Instruction: {prompt}\nOutput:", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"回應: {response}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()