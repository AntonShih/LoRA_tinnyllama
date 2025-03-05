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
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset

# def load_base_model(model_path, use_half_precision=True):
#     """載入基礎模型和tokenizer"""
#     print(f"正在載入基礎模型: {model_path}")
    
#     # 設置模型載入參數
#     dtype = torch.float16 if use_half_precision else torch.float32
    
#     # 載入tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # 載入模型
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=dtype,
#         device_map="auto"
#     )
    
#     return model, tokenizer

def setup_lora_model(base_model, r=8, alpha=16, dropout=0.1, target_modules=None):
    """設置LoRA配置並應用到模型"""
    print("設置LoRA配置...")
    
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    # 配置LoRA參數
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 準備模型並應用LoRA
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, lora_config)
    
    # 輸出可訓練參數的數量
    model.print_trainable_parameters()
    
    return model

def train_lora_model(model, tokenizer, dataset, output_dir, 
                     batch_size=4, grad_accum_steps=4, epochs=3, 
                     learning_rate=2e-4, use_half_precision=True):
    """訓練使用LoRA配置的模型"""
    print("開始LoRA微調...")
    
    # 設定訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=200,
        warmup_ratio=0.03,
        fp16=use_half_precision,
        report_to="tensorboard"
    )
    
    # 設定資料整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 創建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    # 開始訓練
    trainer.train()
    
    # 保存LoRA模型
    lora_output_dir = os.path.join(output_dir, "lora-weights")
    os.makedirs(lora_output_dir, exist_ok=True)
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    
    print(f"LoRA模型已保存至: {lora_output_dir}")
    
    return model, lora_output_dir

def test_model(model, tokenizer, test_prompts):
    """測試模型的生成能力"""
    print("測試模型回應...")
    
    results = []
    for prompt in test_prompts:
        print(f"\n問題: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除可能包含的原始提示
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
            
        print(f"回應: {response}")
        results.append({"prompt": prompt, "response": response})
    
    return results

def load_and_merge_lora_model(base_model_path, lora_weights_path, save_path=None):
    """載入LoRA模型並選擇性地合併"""
    print("載入並合併LoRA模型...")
    
    # 載入基礎模型和tokenizer
    base_model, tokenizer = load_base_model(base_model_path)
    
    # 載入LoRA權重
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    
    if save_path:
        # 合併基礎模型和LoRA權重
        print("正在合併模型...")
        merged_model = model.merge_and_unload()
        
        # 保存合併後的模型
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"合併後的模型已保存至: {save_path}")
        
        return merged_model, tokenizer
    
    return model, tokenizer

def main(base_model_path, instruction_dataset, output_dir, test_prompts=None):
    """執行完整的LoRA微調流程"""
    # 1. 載入基礎模型和tokenizer
    base_model, tokenizer = load_base_model(base_model_path)
    
    # 2. 設置LoRA模型
    lora_model = setup_lora_model(base_model)
    
    # 3. 訓練模型
    trained_model, lora_weights_path = train_lora_model(
        lora_model, tokenizer, instruction_dataset, output_dir
    )
    
    # 4. 測試模型
    if test_prompts:
        test_results = test_model(trained_model, tokenizer, test_prompts)
    
    # 5. 合併和保存最終模型
    merged_model_path = os.path.join(output_dir, "merged-model")
    merged_model, tokenizer = load_and_merge_lora_model(
        base_model_path, lora_weights_path, merged_model_path
    )
    
    print("LoRA微調流程完成!")
    return merged_model, tokenizer, lora_weights_path

# 使用範例
if __name__ == "__main__":
    # 基礎模型路徑
    base_model_path = "./data/model/tinyllama-local"
    
    # 假設你已經準備好了指令資料集
    # instruction_dataset = ...
    
    # 設定輸出目錄
    output_dir = "./results"
    
    # 測試提示
    test_prompts = [
        "我的鞋子上有黑色黴菌，該怎麼處理？",
        "如何防止鞋子發霉？",
        "運動鞋發霉有什麼危害？"
    ]
    
    # 執行完整流程
    model, tokenizer, lora_path = main(
        base_model_path, 
        instruction_dataset,
        output_dir,
        test_prompts
    )