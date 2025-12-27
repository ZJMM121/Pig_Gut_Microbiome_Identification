import argparse
import os
import csv
from collections import defaultdict

def load_id_name_map(file_path):
    """从CSV文件加载ID到名称的映射。"""
    id_to_name = {}
    if not os.path.exists(file_path):
        print(f"错误: 未找到映射文件: {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # 跳过标题行
            if header != ['Global_ID', 'Name']:
                print(f"警告: 映射文件 {file_path} 的标题行不符合预期 ('Global_ID', 'Name')。")
            for row in reader:
                if len(row) == 2:
                    try:
                        global_id = int(row[0])
                        name = row[1]
                        id_to_name[global_id] = name
                    except ValueError:
                        print(f"警告: 映射文件 {file_path} 中有无效行 (非整数ID或格式错误): {row}")
                        continue
                else:
                    print(f"警告: 映射文件 {file_path} 中有格式错误的行 (期望2列): {row}")
                    continue
        print(f"已从 {file_path} 加载 {len(id_to_name)} 个 ID 到名称的映射。")
        return id_to_name
    except Exception as e:
        print(f"加载映射文件 {file_path} 时发生错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='将预测结果中的ID转换为名称，并保存为CSV文件。')
    parser.add_argument('--bacteria_map_path', type=str, 
                        default='./generated_maps/bacteria_id_name_map.csv',
                        help='细菌ID到名称映射CSV文件的路径。')
    parser.add_argument('--trait_map_path', type=str, 
                        default='./generated_maps/trait_id_name_map.csv',
                        help='性状ID到名称映射CSV文件的路径。')
    parser.add_argument('--input_txt', type=str, 
                        required=True,
                        help='包含ID和分数的输入预测 .txt 文件的路径。')
    parser.add_argument('--output_csv', type=str, 
                        required=True,
                        help='输出 .csv 文件的路径 (包含 ID, Name, Score 三列)。')

    args = parser.parse_args()

    # 加载细菌和性状的ID到名称映射
    bacteria_id_to_name = load_id_name_map(args.bacteria_map_path)
    trait_id_to_name = load_id_name_map(args.trait_map_path)

    if bacteria_id_to_name is None:
        print("错误: 无法加载细菌ID到名称的映射。程序退出。")
        return
    # 性状映射不是必需的，如果预测结果中没有性状ID的显示，或者不需要转换性状名称，可以不强制要求其存在。
    # 这里我们允许 trait_id_to_name 为 None，如果它未被加载成功。

    # 用于存储最终结果
    results = []
    
    # 追踪当前处理的性状ID，以便在结果中关联性状名称
    current_trait_id = None 
    current_trait_name = "Unknown_Trait"

    print(f"正在处理输入文件: {args.input_txt}")
    try:
        with open(args.input_txt, 'r', encoding='utf-8') as f_in:
            data_started = False
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                if "新细菌ID" in line and "关联分数" in line: # 匹配表格标题行
                    data_started = True
                    continue
                if data_started and line.startswith("---"): # 匹配分隔线
                    continue
                
                if data_started and line.startswith("性状ID:"):
                    # 尝试从行中提取性状ID
                    parts = line.split("性状ID:")
                    if len(parts) > 1:
                        try:
                            current_trait_id = int(parts[1].strip())
                            if trait_id_to_name: # 如果性状映射存在
                                current_trait_name = trait_id_to_name.get(current_trait_id, f"Trait_ID_{current_trait_id}")
                            else:
                                current_trait_name = f"Trait_ID_{current_trait_id}"
                            print(f"正在处理针对性状: {current_trait_name} (ID: {current_trait_id}) 的预测结果。")
                        except ValueError:
                            print(f"警告: 无法解析性状ID: {line}")
                            current_trait_id = None # 重置为未知
                            current_trait_name = "Unknown_Trait"
                    continue

                if data_started:
                    parts = line.split()
                    if len(parts) == 2:
                        try:
                            entity_id = int(parts[0])
                            score = float(parts[1])
                            
                            # 获取细菌名称
                            bacteria_name = bacteria_id_to_name.get(entity_id, f"Unknown_Bacteria_ID_{entity_id}")
                            
                            results.append({
                                'Trait_ID': current_trait_id,       # 新增性状ID
                                'Trait_Name': current_trait_name,   # 新增性状名称
                                'Bacteria_ID': entity_id,
                                'Bacteria_Name': bacteria_name,
                                'Score': score
                            })
                        except ValueError:
                            print(f"警告: 跳过输入文件中的无效行 (ID或分数无法解析): {line}")
                            continue
                    else:
                        print(f"警告: 跳过输入文件中的格式错误行 (期望ID和分数两列): {line}")
                        continue
    except FileNotFoundError:
        print(f"错误: 未找到输入 TXT 文件: {args.input_txt}。")
        return
    except Exception as e:
        print(f"读取输入 TXT 文件时发生错误: {e}")
        return

    # --- 保存为 .csv 文件 ---
    print(f"正在保存结果到: {args.output_csv}")
    if results:
        # 写入CSV，包含Trait信息
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            # CSV header
            writer.writerow(['Trait_ID', 'Trait_Name', 'Bacteria_ID', 'Bacteria_Name', 'Score']) 
            for row in results:
                writer.writerow([row['Trait_ID'], row['Trait_Name'], row['Bacteria_ID'], row['Bacteria_Name'], row['Score']])
        print("转换完成。CSV 文件已保存。")
    else:
        print("输入文件中没有找到有效数据进行处理。")

if __name__ == '__main__':
    main()