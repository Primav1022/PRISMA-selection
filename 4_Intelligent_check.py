import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
import logging
from pathlib import Path
from tqdm import tqdm  # 添加进度条

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/intelligent_check.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def preprocess_title(title):
    """
    预处理标题文本
    """
    if not isinstance(title, str):
        return ""
    
    # 转换为小写
    title = title.lower()
    
    # 移除标点符号和特殊字符
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # 标准化空格
    title = ' '.join(title.split())
    
    return title

def calculate_similarity(title1, title2):
    """
    计算两个标题的相似度
    只使用序列匹配，移除TF-IDF计算
    """
    # 预处理标题
    title1 = preprocess_title(title1)
    title2 = preprocess_title(title2)
    
    # 如果预处理后为空，返回0
    if not title1 or not title2:
        return 0
    
    # 使用序列匹配相似度
    return SequenceMatcher(None, title1, title2).ratio()

def find_similar_titles(df, similarity_threshold=0.95):
    """
    查找相似标题
    确保每篇论文都被比较到
    """
    similar_pairs = []
    titles = df['title'].tolist()
    total_titles = len(titles)
    
    # 预处理所有标题
    processed_titles = [preprocess_title(title) for title in titles]
    
    # 记录已处理的标题索引
    processed_indices = set()
    
    # 记录比较次数
    comparison_count = 0
    
    # 使用tqdm显示进度条
    for i in tqdm(range(len(titles)), desc="比较标题"):
        # 如果当前标题已处理过，跳过
        if i in processed_indices:
            continue
            
        for j in range(i + 1, len(titles)):
            # 如果j已处理过，跳过
            if j in processed_indices:
                continue
                
            # 如果两个标题长度差异太大，跳过比较
            if abs(len(processed_titles[i]) - len(processed_titles[j])) > 20:
                continue
                
            comparison_count += 1
            similarity = calculate_similarity(titles[i], titles[j])
            
            # 记录每次比较
            logging.info(f"\n比较 #{comparison_count}")
            logging.info(f"标题 {i+1}/{total_titles}: {titles[i]}")
            logging.info(f"标题 {j+1}/{total_titles}: {titles[j]}")
            logging.info(f"相似度: {similarity:.2f}")
            
            if similarity >= similarity_threshold:
                similar_pairs.append({
                    'title1': titles[i],
                    'title2': titles[j],
                    'similarity': similarity
                })
                logging.info(f"✓ 发现相似标题对！")
                # 标记这两个标题为已处理
                processed_indices.add(i)
                processed_indices.add(j)
            else:
                logging.info("✗ 不相似")
    
    # 输出统计信息
    logging.info(f"\n=== 统计信息 ===")
    logging.info(f"总标题数: {total_titles}")
    logging.info(f"实际比较次数: {comparison_count}")
    logging.info(f"发现的相似标题对: {len(similar_pairs)}")
    logging.info(f"跳过的标题数: {len(processed_indices)}")
    
    return similar_pairs

def intelligent_deduplicate(input_file: str, output_file: str = None, similarity_threshold: float = 0.95):
    """
    智能查重并删除重复项
    """
    try:
        # 读取CSV文件
        logging.info(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        
        # 记录原始行数
        original_rows = len(df)
        
        # 检查标题列是否存在
        if 'title' not in df.columns:
            raise ValueError("CSV文件中没有找到'title'列")
        
        # 查找相似标题
        logging.info("\n=== 开始查找相似标题 ===")
        similar_pairs = find_similar_titles(df, similarity_threshold)
        
        if similar_pairs:
            logging.info(f"\n=== 发现的相似标题对 ===")
            for idx, pair in enumerate(similar_pairs, 1):
                logging.info(f"\n相似标题对 #{idx}")
                logging.info(f"相似度: {pair['similarity']:.2f}")
                logging.info(f"标题1: {pair['title1']}")
                logging.info(f"标题2: {pair['title2']}")
            
            # 创建要删除的索引列表
            to_drop = set()
            for pair in similar_pairs:
                # 找到对应的索引
                idx1 = df[df['title'] == pair['title1']].index[0]
                idx2 = df[df['title'] == pair['title2']].index[0]
                # 保留第一个，删除第二个
                to_drop.add(idx2)
            
            # 删除重复项
            df_cleaned = df.drop(index=list(to_drop))
            
            # 确定输出文件路径
            if output_file is None:
                output_file = f"output/{Path(input_file).stem}_intelligent_cleaned.csv"
            
            # 保存结果
            df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 记录处理结果
            removed_rows = original_rows - len(df_cleaned)
            logging.info(f"\n=== 处理结果 ===")
            logging.info(f"原始记录数: {original_rows}")
            logging.info(f"删除的重复记录数: {removed_rows}")
            logging.info(f"保留的记录数: {len(df_cleaned)}")
            logging.info(f"结果已保存到: {output_file}")
        else:
            logging.info("未发现相似标题")
            
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        raise

def main():
    """主函数"""
    input_file = "output/merged_cleaned_results.csv"
    
    if not Path(input_file).exists():
        logging.error(f"找不到文件: {input_file}")
        return
    
    # 设置相似度阈值（可以根据需要调整）
    similarity_threshold = 0.85
    
    # 执行智能查重
    intelligent_deduplicate(input_file, similarity_threshold=similarity_threshold)

if __name__ == "__main__":
    main()