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
from data.training_data.basic_training_data import basic_training_data # 讀取你的原始數據

def convert_to_instruction_format(data):
    """確保每個數據點都是獨立的字典"""
    instruction_data = []
    
    for item in data:
        instruction_data.append({
            "instruction": item["question"],  # ✅ 確保這裡不是列表
            "input": "",  # ✅ 空白輸入
            "output": item["answer"],  # ✅ 確保這裡不是列表
            "category": item["category"]  # ✅ 單獨的分類
        })

    return instruction_data

# 轉換數據
instruction_data = convert_to_instruction_format(basic_training_data)

# 轉換成 Hugging Face Dataset 格式
instruction_dataset = Dataset.from_list(instruction_data)


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


print(f"資料集大小: {len(instruction_dataset)}")
print(instruction_dataset[:2])  # 顯示前 5 筆資料

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
    finally:
        # 釋放資源
        if 'model' in locals():
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()