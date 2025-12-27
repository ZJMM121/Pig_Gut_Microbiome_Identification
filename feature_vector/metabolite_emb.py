import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def smiles_to_morgan(smiles, radius=2, nBits=1024):
    """
    将SMILES字符串转换为Morgan分子指纹
    
    Parameters:
    smiles: SMILES字符串
    radius: 指纹半径，默认2
    nBits: 指纹位数，默认1024
    
    Returns:
    Morgan分子指纹对象，如果失败返回None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    except Exception as e:
        print(f"  SMILES转换失败: {smiles[:50]}... 错误: {e}")
        return None
    return None

# ==== 主函数 ====
def build_metabolite_features_from_csv(input_csv, output_npy, output_names_txt, output_stats_csv=None):
    """
    从包含Compound,CID,SMILES三列的CSV文件中读取数据并生成分子指纹特征
    
    Parameters:
    input_csv: 输入CSV文件路径，包含Compound,CID,SMILES三列
    output_npy: 输出的numpy特征矩阵文件
    output_names_txt: 输出的化合物名称文件
    output_stats_csv: 可选，输出处理统计信息的CSV文件
    """
    print(f"正在读取CSV文件: {input_csv}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(input_csv)
        print(f"成功读取CSV文件，总行数: {len(df)}")
        
        # 检查必需的列
        required_cols = ['Compound', 'CID', 'SMILES']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"错误：CSV文件缺少必需的列: {missing_cols}")
            print(f"实际列名: {list(df.columns)}")
            return
            
        print(f"CSV文件列名: {list(df.columns)}")
        
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 过滤有效数据
    print("正在过滤有效数据...")
    
    # 去除SMILES为空或'Not Found'的行
    valid_df = df[
        (df['SMILES'].notna()) & 
        (df['SMILES'] != 'Not Found') & 
        (df['SMILES'] != '') &
        (df['Compound'].notna()) &
        (df['Compound'] != '')
    ].copy()
    
    print(f"有效数据行数: {len(valid_df)} / {len(df)}")
    print(f"前5个有效化合物:")
    for idx, row in valid_df.head().iterrows():
        print(f"  {row['Compound']} -> {row['SMILES'][:50]}{'...' if len(str(row['SMILES'])) > 50 else ''}")
    
    feature_list = []
    name_list = []
    stats_list = []
    
    print(f"\n开始处理 {len(valid_df)} 个化合物...")
    
    # 处理每个化合物
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="生成分子指纹"):
        compound_name = row['Compound']
        smiles = row['SMILES']
        cid = row['CID']
        
        try:
            # 生成分子指纹
            fp = smiles_to_morgan(smiles)
            
            if fp is not None:
                # 转换为numpy数组
                arr = np.zeros((1024,), dtype=np.int8)
                AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                feature_list.append(arr)
                name_list.append(compound_name)
                
                # 记录统计信息
                stats_list.append({
                    'Compound': compound_name,
                    'CID': cid,
                    'SMILES': smiles,
                    'Status': 'Success',
                    'Error': ''
                })
                
            else:
                print(f"  跳过 {compound_name}: 无法生成分子指纹")
                stats_list.append({
                    'Compound': compound_name,
                    'CID': cid,
                    'SMILES': smiles,
                    'Status': 'Failed',
                    'Error': 'Cannot generate fingerprint'
                })
                
        except Exception as e:
            print(f"  处理 {compound_name} 时出错: {e}")
            stats_list.append({
                'Compound': compound_name,
                'CID': cid,
                'SMILES': smiles,
                'Status': 'Error',
                'Error': str(e)
            })
    
    # 保存结果
    if len(feature_list) > 0:
        print(f"\n成功处理 {len(feature_list)} 个化合物")
        
        # 保存特征矩阵
        feature_matrix = np.array(feature_list)
        np.save(output_npy, feature_matrix)
        print(f"特征矩阵保存到: {output_npy}")
        print(f"特征矩阵形状: {feature_matrix.shape}")
        
        # 保存化合物名称
        with open(output_names_txt, 'w', encoding='utf-8') as f:
            for name in name_list:
                f.write(name + '\n')
        print(f"化合物名称保存到: {output_names_txt}")
        
        # 保存统计信息
        if output_stats_csv:
            stats_df = pd.DataFrame(stats_list)
            stats_df.to_csv(output_stats_csv, index=False)
            print(f"处理统计信息保存到: {output_stats_csv}")
            
            # 打印统计摘要
            success_count = len(stats_df[stats_df['Status'] == 'Success'])
            failed_count = len(stats_df[stats_df['Status'] == 'Failed'])
            error_count = len(stats_df[stats_df['Status'] == 'Error'])
            
            print(f"\n处理摘要:")
            print(f"  成功: {success_count}")
            print(f"  失败: {failed_count}")
            print(f"  错误: {error_count}")
            print(f"  总计: {len(stats_df)}")
            print(f"  成功率: {success_count/len(stats_df)*100:.1f}%")
        
    else:
        print("没有成功处理任何化合物")
        print("请检查输入数据的SMILES格式是否正确")

# ==== 示例调用 ====
if __name__ == "__main__":
    build_metabolite_features_from_csv(
        input_csv="compound_smiles_use.csv",  # 输入CSV文件，包含Compound,CID,SMILES三列
        output_npy="./metabolite_features0725.npy",      # 1024维指纹特征
        output_names_txt="./metabolite_names0725.txt",   # 化合物名称文件
        output_stats_csv="./metabolite_processing_stats0725.csv"  # 处理统计信息
    )