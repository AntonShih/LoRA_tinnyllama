from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)

def load_base_model(model_path):
    """載入基礎模型和tokenizer

    return :
        model, tokenizer
    """
    print(f"正在載入基礎模型: {model_path}")
    
    # 載入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu"
    )
    
    return model, tokenizer