import argparse
import pickle
import os
import csv
from collections import defaultdict

def generate_id_name_maps(metadata_path, original_bacteria_names_path, new_bacteria_names_path, output_dir, trait_names_path=None):
    """
    根据 metadata.pkl 和名称文件生成 Bacteria 和 Trait 的全局ID到名称的映射文件。
    """
    print(f"--- 正在生成 ID 到名称的映射文件 ---")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载 metadata.pkl ---
    print(f"正在从: {metadata_path} 加载元数据...")
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        node_type_map = metadata['node_type_map']
        node_shifts = metadata['node_shifts']
        node_counts = metadata['node_counts']
        print("元数据加载成功。")
    except FileNotFoundError:
        print(f"错误: 未找到 metadata.pkl 文件，路径为 {metadata_path}。请检查路径。")
        return
    except Exception as e:
        print(f"加载 metadata.pkl 时发生错误: {e}")
        return

    # --- 2. 处理 Bacteria 节点的 ID 到名称映射 ---
    bacteria_id_to_name = {}
    bacteria_node_type_name = 'Bacteria'
    
    if bacteria_node_type_name in node_type_map:
        bacteria_global_shift = node_shifts[bacteria_node_type_name]
        total_bacteria_count_in_graph = node_counts[bacteria_node_type_name]
        
        # 加载原始细菌名称
        original_bacteria_names = []
        try:
            with open(original_bacteria_names_path, 'r', encoding='utf-8') as f:
                original_bacteria_names = [line.strip() for line in f if line.strip()]
            print(f"已从 {original_bacteria_names_path} 加载 {len(original_bacteria_names)} 个原始细菌名称。")
        except FileNotFoundError:
            print(f"警告: 未找到原始细菌名称文件 {original_bacteria_names_path}。将跳过原始细菌名称加载。")
        except Exception as e:
            print(f"加载原始细菌名称时发生错误: {e}。将跳过原始细菌名称加载。")

        # 加载新细菌名称
        new_bacteria_names = []
        try:
            with open(new_bacteria_names_path, 'r', encoding='utf-8') as f:
                new_bacteria_names = [line.strip() for line in f if line.strip()]
            print(f"已从 {new_bacteria_names_path} 加载 {len(new_bacteria_names)} 个新细菌名称。")
        except FileNotFoundError:
            print(f"错误: 未找到新细菌名称文件 {new_bacteria_names_path}。这对于构建完整的细菌名称映射至关重要。")
            return
        except Exception as e:
            print(f"加载新细菌名称时发生错误: {e}。")
            return

        # 合并所有细菌名称，并检查总数是否匹配
        all_bacteria_names_ordered = original_bacteria_names + new_bacteria_names
        if len(all_bacteria_names_ordered) != total_bacteria_count_in_graph:
            print(f"错误: 合并后的细菌名称总数 ({len(all_bacteria_names_ordered)}) 与 metadata.pkl 中记录的总细菌数 ({total_bacteria_count_in_graph}) 不匹配。")
            print("请检查您的名称文件，确保它们包含所有细菌且顺序正确。")
            return
        
        # 构建所有 Bacteria 的 ID 到名称映射
        # 假设 HGB 格式的 Bacteria 节点ID是按照 [原始细菌, 新细菌] 的顺序排列的。
        for i in range(total_bacteria_count_in_graph):
            global_id = bacteria_global_shift + i
            bacteria_id_to_name[global_id] = all_bacteria_names_ordered[i]
        
        print(f"已为所有 {total_bacteria_count_in_graph} 个细菌节点构建 ID 到名称的映射。")
        
        # 保存 Bacteria ID-Name 映射到 CSV
        bacteria_output_path = os.path.join(output_dir, 'bacteria_id_name_map.csv')
        with open(bacteria_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Global_ID', 'Name'])
            for global_id in sorted(bacteria_id_to_name.keys()):
                writer.writerow([global_id, bacteria_id_to_name[global_id]])
        print(f"细菌 ID 到名称映射已保存到: {bacteria_output_path}")

    else:
        print(f"警告: 元数据中未找到 '{bacteria_node_type_name}' 节点类型。跳过细菌映射生成。")

    # --- 3. 处理 Trait 节点的 ID 到名称映射 ---
    trait_id_to_name = {}
    trait_node_type_name = 'Trait'

    if trait_node_type_name in node_type_map:
        trait_global_shift = node_shifts[trait_node_type_name]
        total_trait_count = node_counts[trait_node_type_name]
        
        trait_names = []
        if trait_names_path and os.path.exists(trait_names_path):
            try:
                with open(trait_names_path, 'r', encoding='utf-8') as f:
                    trait_names = [line.strip() for line in f if line.strip()]
                print(f"已从 {trait_names_path} 加载 {len(trait_names)} 个性状名称。")
            except Exception as e:
                print(f"警告: 加载性状名称时发生错误: {e}。将使用通用名称。")
                trait_names = [f"Trait_Name_{i+1}" for i in range(total_trait_count)]
        else:
            print(f"警告: 未提供性状名称路径 '{trait_names_path}' 或未找到文件。将使用通用名称。")
            trait_names = [f"Trait_Name_{i+1}" for i in range(total_trait_count)]

        # 检查性状名称数量是否匹配
        if len(trait_names) != total_trait_count:
            print(f"警告: 性状名称文件中的名称数量 ({len(trait_names)}) 与 metadata.pkl 中记录的总性状数 ({total_trait_count}) 不匹配。")
            print("请检查您的性状名称文件。")
            # 即使不匹配也继续，但会使用通用名称或截断/填充
            if len(trait_names) < total_trait_count:
                trait_names.extend([f"Trait_Name_Undefined_{i}" for i in range(total_trait_count - len(trait_names))])
            else:
                trait_names = trait_names[:total_trait_count]

        for i in range(total_trait_count):
            global_id = trait_global_shift + i
            trait_id_to_name[global_id] = trait_names[i]
        
        print(f"已为所有 {total_trait_count} 个性状节点构建 ID 到名称的映射。")

        # 保存 Trait ID-Name 映射到 CSV
        trait_output_path = os.path.join(output_dir, 'trait_id_name_map.csv')
        with open(trait_output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Global_ID', 'Name'])
            for global_id in sorted(trait_id_to_name.keys()):
                writer.writerow([global_id, trait_id_to_name[global_id]])
        print(f"性状 ID 到名称映射已保存到: {trait_output_path}")

    else:
        print(f"警告: 元数据中未找到 '{trait_node_type_name}' 节点类型。跳过性状映射生成。")

    print(f"--- ID 到名称映射生成完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为细菌和性状节点生成 ID 到名称的映射文件。')
    parser.add_argument('--metadata_path', type=str, 
                        default='/user_data/yezy/zhangjm/Simple-HGN-PGMKG2-KD/data/PGMKG_HGB0810/metadata.pkl',
                        help='metadata.pkl 文件的路径。')
    parser.add_argument('--original_bacteria_names_path', type=str, 
                        default='/user_data/yezy/zhangjm/FE_kg/Merge/feature_vector/bacteria_wang_names0724.txt',
                        help='原始细菌名称 .txt 文件的路径。')
    parser.add_argument('--new_bacteria_names_path', type=str, 
                        required=True, # 新细菌名称文件现在是必须的
                        help='新细菌名称 .txt 文件的路径。')
    parser.add_argument('--trait_names_path', type=str, 
                        default=None, 
                        help='可选: 性状名称 .txt 文件的路径。')
    parser.add_argument('--output_dir', type=str, 
                        default='./id_name_maps',
                        help='保存生成的 ID-名称 CSV 文件的目录。')

    args = parser.parse_args()
    
    generate_id_name_maps(args.metadata_path, 
                          args.original_bacteria_names_path, 
                          args.new_bacteria_names_path, # 传递新细菌名称文件路径
                          args.output_dir, 
                          args.trait_names_path)