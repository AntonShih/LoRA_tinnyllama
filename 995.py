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

from data.training_data.basic_training_data import basic_training_data

# 轉換資料和標記函數
def convert_to_instruction_format(data):
    instruction_data = []
    for item in data:
        instruction_item = {
            "instruction": item["question"],
            "input": "",
            "output": item["answer"],
            "category": item["category"]
        }
        instruction_data.append(instruction_item)
    return instruction_data

def tokenize_function(examples, tokenizer, max_length=512):
    prompts = [
        f"Instruction: {instr}\nOutput: {out}"
        for instr, out in zip(examples["instruction"], examples["output"])
    ]
    
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# 主要程序
def main():
    print("開始準備訓練資料...")
    
    instruction_data = convert_to_instruction_format(basic_training_data)
    dataset = Dataset.from_list(instruction_data)

    model_path = "./data/model/tinyllama-local"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 標記化資料
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 載入模型
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": "cpu"}
    )
    
    # 設定 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    # 設定訓練參數
    output_dir = "./results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-4,  # 🟢 **提高學習率，讓小數據集更快收斂**
        per_device_train_batch_size=1,  # ✅ CPU 記憶體受限，batch size 設 1
        gradient_accumulation_steps=1,  # 🔵 **CPU 訓練不需要梯度累積**
        num_train_epochs=10,  # 🟢 **100 筆數據應該增加 Epoch**
        weight_decay=0.01,
        logging_steps=2,  # 🔵 **減少 logging，避免影響 CPU 訓練**
        save_steps=10,
        warmup_ratio=0.1,  # 🟢 **小數據 warmup 多一點，提高穩定性**
        save_total_limit=2,
        no_cuda=True,  # 🟢 **確保只用 CPU**
        fp16=False,  # 🛑 **CPU 不支援 FP16，需關閉**
        report_to="none"
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Early Stopping 條件
    best_loss = float("inf")  # 記錄最低 Loss
    patience = 3  # 允許多少個 Epoch 沒有改善
    counter = 0  # 記錄連續未改善次數

    print("開始訓練...")
    
    for epoch in range(training_args.num_train_epochs):
        train_result = trainer.train(resume_from_checkpoint=False)
        current_loss = train_result.training_loss  # 取得當前 Loss
        
        print(f"Epoch {epoch+1} 損失: {current_loss:.4f}")

        # 檢查 Loss 是否改善
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0  # 重置計數器
        else:
            counter += 1  # 連續未改善 +1

        # 若超過 patience，則提前停止
        if counter >= patience:
            print(f"Loss 未改善 {patience} 個 Epoch，提前停止訓練！")
            break

    # 儲存模型
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
