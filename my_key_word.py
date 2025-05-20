import pandas as pd
import nltk
import re
import sys
from collections import Counter
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple

# 我们可以尝试显式添加已知的 NLTK 数据路径，看看是否有帮助
# 如果下面的诊断显示这个路径不在 nltk.data.path 中，可以取消这行的注释
# nltk.data.path.append('C:\\Users\\李琛子\\AppData\\Roaming\\nltk_data')
# print(f"Manually added to nltk.data.path. Current nltk.data.path: {nltk.data.path}")

# 确保已下载NLTK资源 (如果脚本在不同环境运行，这行可以注释掉，手动下载一次即可)
# try:
#     stopwords.words('english')
#     word_tokenize("test")
#     nltk.pos_tag(word_tokenize("test")) # Test POS tagger
#     WordNetLemmatizer().lemmatize("tests") # Test Lemmatizer
# except LookupError:
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True) # For POS tagging
#     nltk.download('wordnet', quiet=True) # For Lemmatization

# --- 新增代码段开始 ---
# 尝试将 conda 环境的 NLTK 数据路径置于 nltk.data.path 的最前面
# 请根据您的实际 conda 环境路径调整
conda_env_nltk_data_path_primary = 'D:\\miniconda\\envs\\prisma_env\\nltk_data'
conda_env_nltk_data_path_lib = 'D:\\miniconda\\envs\\prisma_env\\lib\\nltk_data' # 有时lib下也有

# 确保路径不重复添加，并加到最前面
if conda_env_nltk_data_path_lib in nltk.data.path:
    nltk.data.path.remove(conda_env_nltk_data_path_lib)
nltk.data.path.insert(0, conda_env_nltk_data_path_lib)

if conda_env_nltk_data_path_primary in nltk.data.path:
    nltk.data.path.remove(conda_env_nltk_data_path_primary)
nltk.data.path.insert(0, conda_env_nltk_data_path_primary)

print(f"修改后优先的 NLTK 搜索路径 (nltk.data.path): {nltk.data.path}")
# --- 新增代码段结束 ---

def get_wordnet_pos(treebank_tag: str):
    """将treebank词性标记转换为WordNet词性标记"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # 默认为名词


def preprocess_text(text: str, language: str = 'english', lemmatizer=None) -> List[str]:
    """
    对单个文本字符串进行预处理：
    1. 转换为小写
    2. 去除非字母数字字符 (保留空格和一些内部连字符，可选)
    3. 分词
    4. 去除停用词
    5. 词性标注并筛选名词和形容词
    6. 词形还原
    Args:
        text: 输入的文本字符串。
        language: 文本的语言 (目前主要支持 'english')。
        lemmatizer: 外部传入的 WordNetLemmatizer 实例。
    Returns:
        预处理后的词语列表 (仅包含名词和形容词的词根)。
    """
    if not isinstance(text, str):
        return []

    text = text.lower() # 转为小写
    # 去除非字母字符，但保留单词内部的连字符和撇号
    text = re.sub(r'[^a-z0-9\\s\-\']', ' ', text) # 保留数字、空格、连字符、撇号
    text = re.sub(r'\\s+', ' ', text).strip() # 合并多个空格

    tokens = word_tokenize(text, language=language)
    
    # 获取停用词列表
    stop_words_list = set(stopwords.words(language))
    
    # 词性标注
    tagged_tokens = nltk.pos_tag(tokens)
    
    # 筛选名词 (NN, NNS, NNP, NNPS) 和形容词 (JJ, JJR, JJS)
    # 同时去除停用词和短词
    filtered_words = []
    for word, tag in tagged_tokens:
        if word not in stop_words_list and len(word) > 2:
            if tag.startswith('NN') or tag.startswith('JJ'): # 名词或形容词
                filtered_words.append((word, get_wordnet_pos(tag))) # 保留词和其WordNet词性
    
    if lemmatizer is None: # 如果没有传入lemmatizer，则创建一个
        lemmatizer = WordNetLemmatizer()
        
    # 词形还原
    lemmatized_words = [lemmatizer.lemmatize(word, pos=pos) for word, pos in filtered_words]
    
    # 再次过滤，去除词形还原后可能产生的停用词或短词 (可选，但有时有用)
    final_words = [
        word for word in lemmatized_words
        if word not in stop_words_list and len(word) > 2 and word.isalpha() # 确保是纯字母
    ]
    
    return final_words

def get_word_frequencies(text_list: List[str], top_n: int = 30) -> List[Tuple[str, int]]:
    """
    计算词频并返回最高频的N个词。
    Args:
        text_list: 包含多个预处理后词语列表的列表。
        top_n: 返回最高频词的数量。
    Returns:
        一个元组列表，每个元组包含 (词语, 频率)。
    """
    all_words = []
    for words in text_list:
        all_words.extend(words)
    
    if not all_words:
        return []
        
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(top_n)
    return most_common_words

def main():
    """主函数，加载数据，进行词频分析并输出结果。"""
    csv_file_path = 'output/scholar_results.csv'  # CSV文件路径
    column_name = 'title' # 需要分析的列名
    num_keywords = 30 # 希望得到的关键词数量

    # 打印一下nltk.data.path，确保我们的修改生效了
    print(f"NLTK 将从以下路径查找数据: {nltk.data.path}")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"错误: CSV文件 '{csv_file_path}' 未找到。请确保文件存在于脚本同目录下，或提供正确路径。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
        return

    if column_name not in df.columns:
        print(f"错误: 列 '{column_name}' 在CSV文件中未找到。可用列: {df.columns.tolist()}")
        return

    # 获取标题列表，并处理可能的非字符串类型（如NaN）
    titles = df[column_name].astype(str).tolist()
    
    print(f"正在从 '{csv_file_path}' 的 '{column_name}' 列处理 {len(titles)} 个标题...")

    # 初始化词形还原器以便复用
    lemmatizer = WordNetLemmatizer()

    processed_titles_tokens = []
    for title_text in titles:
        processed_titles_tokens.append(preprocess_text(title_text, lemmatizer=lemmatizer))
    
    print(f"\n筛选并词形还原后，词频最高的 {num_keywords} 个名词/形容词:")
    top_words = get_word_frequencies(processed_titles_tokens, top_n=num_keywords)
    
    if top_words:
        for i, (word, freq) in enumerate(top_words):
            print(f"{i+1}. {word}: {freq}")
    else:
        print("未能计算词频，可能是因为文本预处理后没有有效词语或所有标题均为空。")

if __name__ == "__main__":
    main()
