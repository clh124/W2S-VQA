import json

def filter_json(input_file, output_filtered, output_remaining):
    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = []  # 存放包含 'test_add' 的数据
    remaining_data = []  # 存放剩余数据
    
    for item in data:
        if any("test_add" in image for image in item.get("image", [])):
            filtered_data.append(item)
        else:
            remaining_data.append(item)
    
    # 保存包含 'test_add' 的数据
    with open(output_filtered, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    # 保存剩余数据
    with open(output_remaining, 'w', encoding='utf-8') as f:
        json.dump(remaining_data, f, indent=4, ensure_ascii=False)
    
    print(f"Filtered JSON saved to {output_filtered}")
    print(f"Remaining JSON saved to {output_remaining}")

# 示例调用
input_json = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/train_3w_stage2_ft_total.json"  # 输入 JSON 文件路径
output_filtered_json = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/train_3w_stage2_ft_test_add.json"  # 存放筛选出的 JSON
output_remaining_json = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/train_3w_stage2_ft_remaining.json"  # 存放剩余 JSON

# filter_json(input_json, output_filtered_json, output_remaining_json)


import os
import json
import random

def merge_and_shuffle_json(input_folder, output_file):
    all_data = []

    # 遍历文件夹中的所有 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):  # 确保 JSON 文件内容是列表
                    all_data.extend(data)
                else:
                    all_data.append(data)  # 处理 JSON 结构为字典的情况

    # 随机打乱数据顺序
    random.shuffle(all_data)

    # 保存合并后的 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"Merged JSON saved to {output_file}, total {len(all_data)} entries.")

# 使用示例
input_folder = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/"  # 替换为你的 JSON 文件夹路径
output_file = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/stage2_merged_shuffled_3w.json"
# merge_and_shuffle_json(input_folder, output_file)


import json
import random

def process_json_files(base_json, extra_json, output_file, target_count=30000, copies=20):
    # 读取第一个 JSON 文件
    with open(base_json, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # 复制 10 次
    base_data_extended = base_data * copies
    base_total = len(base_data_extended)
    print(f"Base dataset after {copies} copies: {base_total} entries.")

    # 读取第二个 JSON 文件
    with open(extra_json, "r", encoding="utf-8") as f:
        extra_data = json.load(f)

    # 计算需要补充的元素数量
    required_extra = target_count - base_total
    if required_extra <= 0:
        print(f"Base dataset already exceeds {target_count}, trimming instead.")
        final_data = random.sample(base_data_extended, target_count)
    else:
        print(f"Selecting {required_extra} elements from extra dataset.")
        extra_selected = random.sample(extra_data, required_extra) if required_extra < len(extra_data) else extra_data
        final_data = base_data_extended + extra_selected

    # 随机打乱数据
    random.shuffle(final_data)

    # 保存最终 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"Final dataset saved to {output_file}, total {len(final_data)} entries.")

# 使用示例
base_json = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/train_3w_stage2_ft_test_add.json"  # 需要复制 10 次的 JSON 文件
extra_json = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/train_3w_stage2_ft_remaining.json"  # 用于补充的 JSON 文件
output_file = "/root/workspace/cusvgvkp420c73amv8l0/code/llm_pair_vqa/llava/stage2/train_3w_stage2_ft_total_3w.json"  # 输出文件
process_json_files(base_json, extra_json, output_file)