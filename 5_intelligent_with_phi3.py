#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 论文智能分析系统 - 使用 Phi-3-mini 模型
==================================================

这个脚本用于：
1. 从 CSV 文件中读取论文信息
2. 下载或处理 PDF 文件
3. 使用 Phi-3-mini 模型进行文本总结和分析
4. 生成可视化结果和报告

作者: 
日期: 2024
版本: 1.0
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import fitz  # PyMuPDF
import pdfplumber
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
import time

# AI 模型相关
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 进度条
from tqdm.auto import tqdm

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PDFProcessor:
    """PDF 文件处理器"""
    
    def __init__(self, download_dir: str = "pdfs"):
        """
        初始化 PDF 处理器
        
        Args:
            download_dir: PDF 文件下载目录
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('log/pdf_processing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def download_pdf(self, url: str, title: str) -> Optional[str]:
        """
        下载 PDF 文件
        
        Args:
            url: PDF 文件 URL
            title: 论文标题（用于文件命名）
            
        Returns:
            下载的文件路径，失败返回 None
        """
        try:
            # 清理文件名
            safe_title = re.sub(r'[<>:"/\\|?*]', '', title)[:100]
            filename = f"{safe_title}.pdf"
            filepath = self.download_dir / filename
            
            # 如果文件已存在，直接返回
            if filepath.exists():
                self.logger.info(f"文件已存在: {filename}")
                return str(filepath)
            
            # 下载文件
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 检查是否为 PDF 文件
            if 'application/pdf' not in response.headers.get('content-type', ''):
                self.logger.warning(f"URL 不是 PDF 文件: {url}")
                return None
            
            # 保存文件
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"成功下载: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"下载失败 {url}: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """
        从 PDF 文件中提取文本
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            提取的文本内容，失败返回 None
        """
        try:
            text_content = ""
            
            # 方法1: 使用 PyMuPDF
            try:
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text_content += page.get_text()
                        
                if len(text_content.strip()) > 100:  # 如果提取到足够的文本
                    return text_content
            except Exception as e:
                self.logger.warning(f"PyMuPDF 提取失败: {str(e)}")
            
            # 方法2: 使用 pdfplumber（备用方案）
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                            
                if len(text_content.strip()) > 100:
                    return text_content
            except Exception as e:
                self.logger.warning(f"pdfplumber 提取失败: {str(e)}")
            
            return text_content if text_content.strip() else None
            
        except Exception as e:
            self.logger.error(f"文本提取失败 {pdf_path}: {str(e)}")
            return None
    
    def preprocess_text(self, text: str, max_length: int = 4000) -> str:
        """
        预处理文本内容
        
        Args:
            text: 原始文本
            max_length: 最大长度限制
            
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 清理文本
        text = re.sub(r'\s+', ' ', text)  # 合并多个空白字符
        text = re.sub(r'\n+', '\n', text)  # 合并多个换行符
        text = text.strip()
        
        # 截取前面部分（通常包含摘要和主要内容）
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text


class Phi3Analyzer:
    """Phi-3 模型分析器"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        """
        初始化 Phi-3 分析器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            self.logger.info(f"正在加载模型: {self.model_name}")
            self.logger.info(f"使用设备: {self.device}")
            
            # 尝试多次下载，处理网络问题
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # 加载分词器
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        padding_side="left",
                        force_download=(attempt > 0)  # 第一次失败后强制重新下载
                    )
                    
                    # 设置 pad_token
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # 加载模型
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None,
                        trust_remote_code=True,
                        force_download=(attempt > 0),  # 第一次失败后强制重新下载
                        attn_implementation="eager"  # 使用eager attention，避免flash_attn警告
                    )
                    
                    if self.device.type == "cpu":
                        self.model = self.model.to(self.device)
                    
                    self.logger.info("模型加载成功")
                    return  # 成功后退出重试循环
                    
                except Exception as e:
                    self.logger.warning(f"第 {attempt + 1} 次尝试失败: {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # 最后一次尝试失败时抛出异常
                    
                    # 清理可能损坏的缓存文件
                    import shutil
                    from pathlib import Path
                    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_name.replace('/', '--')}"
                    if cache_dir.exists():
                        self.logger.info("清理可能损坏的缓存文件...")
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    
                    time.sleep(2)  # 等待2秒后重试
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            self.logger.info("建议解决方案:")
            self.logger.info("1. 检查网络连接")
            self.logger.info("2. 设置 HuggingFace 镜像: export HF_ENDPOINT=https://hf-mirror.com")
            self.logger.info("3. 手动下载模型到本地")
            raise
    
    def generate_summary(self, text: str, max_new_tokens: int = 256) -> str:
        """
        生成文本摘要
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成令牌数
            
        Returns:
            生成的摘要
        """
        try:
            # 构建提示词
            prompt = f"""请为以下学术论文内容生成一个简洁的中文摘要，包含以下要点：
1. 研究主题和目标
2. 主要方法或技术
3. 关键发现或贡献
4. 应用价值或意义

论文内容：
{text}

摘要："""

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500,  # 为生成留出空间
                padding=True
            ).to(self.device)
            
            # 生成摘要
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码输出
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"摘要生成失败: {str(e)}")
            return f"摘要生成失败: {str(e)}"
    
    def extract_keywords(self, text: str, max_new_tokens: int = 100) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            max_new_tokens: 最大生成令牌数
            
        Returns:
            关键词列表
        """
        try:
            prompt = f"""请从以下学术论文内容中提取5-10个最重要的关键词，用逗号分隔：

论文内容：
{text[:2000]}

关键词："""

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2500,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # 解析关键词
            keywords = [kw.strip() for kw in generated_text.split(',')]
            return [kw for kw in keywords if kw and len(kw) > 1][:10]
            
        except Exception as e:
            self.logger.error(f"关键词提取失败: {str(e)}")
            return []


def main():
    """主函数"""
    try:
        print("🚀 启动论文智能分析系统")
        print("📋 功能：")
        print("  1. PDF 文本提取")
        print("  2. Phi-3-mini 模型分析")
        print("  3. 智能摘要生成")
        print("  4. 关键词提取")
        print("-" * 50)
        
        # 示例用法
        csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
        
        if not os.path.exists(csv_file):
            print(f"❌ 错误: 找不到文件 {csv_file}")
            print("请确保 CSV 文件存在")
            return
        
        # 读取数据
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"📊 成功加载 {len(df)} 条论文记录")
        
        # 初始化处理器
        pdf_processor = PDFProcessor()
        phi3_analyzer = Phi3Analyzer()
        
        print("\n✅ 系统初始化完成！")
        print("💡 提示：这是一个示例脚本，您可以根据需要修改和扩展功能。")
        
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
