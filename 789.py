import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    PeftModel
)

from datasets import Dataset

# 不從 basic_training_data.py 導入，而是直接在這裡定義資料
# 從 data.training_data.basic_training_data import basic_training_data
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
    }
    # 這裡只定義了兩筆資料作為示例，您需要添加其餘資料或正確導入
]

# 檢查原始資料格式
print("原始資料型態:", type(basic_training_data))
print("原始資料第一筆:", basic_training_data[0])
print("原始資料大小:", len(basic_training_data))

def convert_to_instruction_format(data):
    """將原始資料轉換為模型訓練所需的格式"""
    instruction_data = []
    
    for item in data:
        # 為每個資料項創建一個新的字典
        instruction_item = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"],
            "category": item["category"]
        }
        instruction_data.append(instruction_item)
    
    return instruction_data

# 轉換資料
instruction_data = convert_to_instruction_format(basic_training_data)
print("轉換後第一筆資料:", instruction_data[0])
print("轉換後資料大小:", len(instruction_data))

# 轉換成 Hugging Face Dataset 格式
instruction_dataset = Dataset.from_list(instruction_data)
print("Dataset 第一筆:", instruction_dataset[0])
print("Dataset 大小:", len(instruction_dataset))

def load_base_model(model_path):
    """載入基礎模型和 tokenizer（適用於 CPU 訓練）"""
    print(f"正在載入基礎模型: {model_path}")

    # ✅ 確保 CPU 訓練使用 float32，如果支援 bf16 則使用
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 確保 pad token 設定正確

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": "cpu"}  # ✅ 強制使用 CPU
    )

    return model, tokenizer

def setup_lora_model(base_model, r=8, alpha=16, dropout=0.1, target_modules=None):
    """設定 LoRA 並應用到模型"""
    print("設定 LoRA 配置...")

    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # 確保 TinyLlama 相容

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model

def train_lora_model(model, tokenizer, dataset, output_dir):
    """訓練 LoRA 模型（適用於 CPU 訓練）"""
    print("開始 LoRA 微調...")

    # ✅ 確保 CPU 訓練使用 float32，如果支援 bf16 則使用
    bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        warmup_ratio=0.03,
        bf16=bf16,  # ✅ CPU 訓練時使用 bf16（如果支援），否則回退 float32
        report_to="tensorboard"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    # ✅ 儲存 LoRA 權重
    lora_output_dir = os.path.join(output_dir, "lora-weights")
    os.makedirs(lora_output_dir, exist_ok=True)
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)

    print(f"LoRA 模型已保存至: {lora_output_dir}")

    return model, lora_output_dir

def test_model(model, tokenizer, test_prompts):
    """測試 LoRA 模型的生成能力"""
    print("測試模型回應...")

    results = []
    for prompt in test_prompts:
        print(f"\n問題: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}  # ✅ 確保在 CPU 運行

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
        results.append({"prompt": prompt, "response": response})

    return results

def main(base_model_path, instruction_dataset, output_dir, test_prompts=None):
    """執行完整的 LoRA 訓練流程"""
    base_model, tokenizer = load_base_model(base_model_path)
    lora_model = setup_lora_model(base_model)

    trained_model, lora_weights_path = train_lora_model(
        lora_model, tokenizer, instruction_dataset, output_dir
    )

    if test_prompts:
        test_results = test_model(trained_model, tokenizer, test_prompts)

    print("LoRA 微調流程完成!")
    return trained_model, tokenizer, lora_weights_path

if __name__ == "__main__":
    base_model_path = "./data/model/tinyllama-local"
    output_dir = "./results"

    test_prompts = [
        "如何防止鞋子發霉？",
        "鞋子上的黴菌有什麼危害？"
    ]

    try:
        model, tokenizer, lora_path = main(
            base_model_path, 
            instruction_dataset,  # 已在上方定義
            output_dir,
            test_prompts
        )

        print(f"成功完成訓練，LoRA 權重保存在: {lora_path}")
    except Exception as e:
        print(f"訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()  # 輸出完整錯誤堆疊
    finally:
        # 釋放資源
        if 'model' in locals():
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()