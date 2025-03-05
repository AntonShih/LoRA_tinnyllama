from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 下載模型和tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,  # 使用半精度減少記憶體需求
    device_map="cpu"  # 明確指定使用cPU
)

# 可以保存到本地以便重複使用
tokenizer.save_pretrained("./data/model/tinyllama-local")
model.save_pretrained("./data/model/tinyllama-local")