from datasets import Dataset
import pandas as pd

def convert_to_instruction_format(data):
    instruction_data = []
    for item in data:
        # 創建指令格式的資料
        instruction_item = {
            "instruction": item["question"],
            "input": "",  # 一般問答不需要額外輸入
            "output": item["answer"],
            "category": item["category"]  # 保留分類資訊
        }
        instruction_data.append(instruction_item)
    return instruction_data


if __name__ == "__main__":
    # 指定資料集
    import sys
    import os
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    
    from data.training_data.basic_training_data import basic_training_data
    data = basic_training_data

    # 轉換為指令格式
    instruction_data = convert_to_instruction_format(data)
    instruction_dataset = Dataset.from_list(instruction_data)

    # 顯示數據集統計資訊
    print(f"原始資料集大小: {len(data)}")
    print(f"指令格式資料集大小: {len(instruction_dataset)}")
    print("\n指令格式資料集樣本:")
    print(instruction_dataset[0])