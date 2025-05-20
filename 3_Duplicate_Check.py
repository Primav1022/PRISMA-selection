import pandas as pd
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/duplicate_check.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def check_and_remove_duplicates(input_file: str, output_file: str = None) -> None:
    """
    检查并删除重复的论文记录
    
    Args:
        input_file: 输入文件路径（CSV文件）
        output_file: 输出文件路径（如果不指定，将覆盖原文件）
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
        
        # 查找重复项
        duplicates = df[df.duplicated(subset=['title'], keep='first')]
        
        if len(duplicates) > 0:
            logging.info(f"发现 {len(duplicates)} 条重复记录")
            
            # 显示重复的记录
            for idx, row in duplicates.iterrows():
                logging.info(f"重复记录 - 标题: {row['title']}")
            
            # 删除重复项，保留第一条记录
            df_cleaned = df.drop_duplicates(subset=['title'], keep='first')
        else:
            logging.info("未发现重复记录")
            df_cleaned = df  # 即使没有重复，也使用原始数据框
        
        # 确定输出文件路径
        if output_file is None:
            output_file = input_file
        
        # 保存结果
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 记录处理结果
        removed_rows = original_rows - len(df_cleaned)
        logging.info(f"处理完成！")
        logging.info(f"原始记录数: {original_rows}")
        logging.info(f"删除的重复记录数: {removed_rows}")
        logging.info(f"保留的记录数: {len(df_cleaned)}")
        logging.info(f"结果已保存到: {output_file}")
            
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        raise

def merge_and_deduplicate_csv_files(cleaned_files: list, output_file: str = "output/merged_cleaned_results.csv") -> None:
    """
    合并所有清理后的CSV文件并去除重复项
    
    Args:
        cleaned_files: 清理后的CSV文件列表
        output_file: 最终合并后的输出文件名
    """
    try:
        # 存储所有数据框
        dfs = []
        
        # 读取所有清理后的CSV文件
        for file in cleaned_files:
            logging.info(f"正在读取清理后的文件: {file}")
            df = pd.read_csv(file, encoding='utf-8-sig')
            dfs.append(df)
        
        # 合并所有数据框
        merged_df = pd.concat(dfs, ignore_index=True)
        original_rows = len(merged_df)
        
        # 跨文件去重
        merged_df_cleaned = merged_df.drop_duplicates(subset=['title'], keep='first')
        
        # 记录跨文件重复项
        duplicates = merged_df[merged_df.duplicated(subset=['title'], keep='first')]
        if len(duplicates) > 0:
            logging.info(f"\n发现跨文件重复记录 {len(duplicates)} 条")
            for idx, row in duplicates.iterrows():
                logging.info(f"跨文件重复 - 标题: {row['title']}")
        
        # 保存最终结果
        merged_df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 记录处理结果
        removed_rows = original_rows - len(merged_df_cleaned)
        logging.info(f"\n跨文件去重处理完成！")
        logging.info(f"合并后的总记录数: {original_rows}")
        logging.info(f"删除的跨文件重复记录数: {removed_rows}")
        logging.info(f"最终保留的记录数: {len(merged_df_cleaned)}")
        logging.info(f"最终结果已保存到: {output_file}")
        
    except Exception as e:
        logging.error(f"合并文件时出错: {str(e)}")
        raise

def main():
    """主函数"""
    # 获取output文件夹下的所有CSV文件
    output_dir = Path('output')
    if not output_dir.exists():
        logging.error("output文件夹不存在")
        return
        
    csv_files = list(output_dir.glob('*.csv'))
    
    if not csv_files:
        logging.warning("当前目录下没有找到CSV文件")
        return
    
    # 存储清理后的文件路径
    cleaned_files = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            # 为每个文件创建对应的输出文件名
            output_file = f"{csv_file.stem}_cleaned{csv_file.suffix}"
            
            logging.info(f"\n开始处理文件: {csv_file}")
            check_and_remove_duplicates(str(csv_file), output_file)
            
            # 将清理后的文件路径添加到列表中
            cleaned_files.append(output_file)
            
        except Exception as e:
            logging.error(f"处理文件 {csv_file} 时出错: {str(e)}")
            continue
    
    # 如果存在清理后的文件，进行跨文件去重
    if cleaned_files:
        logging.info("\n开始跨文件去重处理...")
        merge_and_deduplicate_csv_files(cleaned_files)

if __name__ == "__main__":
    main()
