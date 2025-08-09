#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例：如何使用 Phi-3 论文分析系统
=====================================

这个文件展示了如何使用 5_intelligent_with_phi3.py 中的各个组件
来分析论文数据并生成智能摘要。

"""

import pandas as pd
import os
from pathlib import Path

# 导入我们的分析模块
# 注意：文件名中的数字开头需要特殊处理
import importlib.util
import sys

def import_phi3_module():
    """动态导入 5_intelligent_with_phi3.py 模块"""
    spec = importlib.util.spec_from_file_location(
        "intelligent_with_phi3", 
        "5_intelligent_with_phi3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["intelligent_with_phi3"] = module
    spec.loader.exec_module(module)
    return module

try:
    phi3_module = import_phi3_module()
    PDFProcessor = phi3_module.PDFProcessor
    Phi3Analyzer = phi3_module.Phi3Analyzer
except Exception as e:
    print(f"❌ 无法导入分析模块: {e}")
    print("请确保 5_intelligent_with_phi3.py 文件存在且所有依赖已安装")
    sys.exit(1)

def example_basic_usage():
    """基础使用示例"""
    print("🔬 基础使用示例")
    print("=" * 50)
    
    # 1. 初始化组件
    print("📝 步骤 1: 初始化组件")
    pdf_processor = PDFProcessor(download_dir="example_pdfs")
    phi3_analyzer = Phi3Analyzer()
    
    # 2. 示例文本分析（使用摘要文本）
    print("\n📝 步骤 2: 分析示例文本")
    sample_text = """
    This paper presents a novel approach to data-driven design for metamaterials 
    and multiscale systems. The research focuses on developing computational methods 
    that can automatically generate and optimize material structures based on desired 
    properties. The methodology combines machine learning algorithms with physics-based 
    simulations to create materials with unprecedented characteristics. The results 
    demonstrate significant improvements in material performance across various 
    applications including aerospace, automotive, and biomedical engineering.
    """
    
    # 生成摘要
    print("🤖 生成 AI 摘要...")
    summary = phi3_analyzer.generate_summary(sample_text)
    print(f"摘要: {summary}")
    
    # 提取关键词
    print("\n🔍 提取关键词...")
    keywords = phi3_analyzer.extract_keywords(sample_text)
    print(f"关键词: {', '.join(keywords)}")

def example_csv_processing():
    """处理 CSV 文件示例"""
    print("\n\n📊 CSV 文件处理示例")
    print("=" * 50)
    
    csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        print("请确保您已经运行了前面的数据收集步骤")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"📈 加载了 {len(df)} 条论文记录")
    
    # 显示前几条记录的信息
    print("\n📋 前 3 条记录的基本信息:")
    for i, row in df.head(3).iterrows():
        print(f"\n论文 {i+1}:")
        print(f"  标题: {row['title'][:80]}...")
        print(f"  作者: {row['authors']}")
        print(f"  年份: {row['year']}")
        print(f"  引用数: {row['citations']}")
        if 'abstract' in row and pd.notna(row['abstract']):
            print(f"  摘要: {row['abstract'][:100]}...")

def example_single_paper_analysis():
    """单篇论文详细分析示例"""
    print("\n\n🔍 单篇论文详细分析示例")
    print("=" * 50)
    
    csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return
    
    # 读取数据并选择一篇论文
    df = pd.read_csv(csv_file, encoding='utf-8')
    paper = df.iloc[0]  # 选择第一篇论文
    
    print(f"📄 分析论文: {paper['title']}")
    print(f"👥 作者: {paper['authors']}")
    print(f"📅 年份: {paper['year']}")
    print(f"📊 引用数: {paper['citations']}")
    
    # 初始化分析器
    pdf_processor = PDFProcessor()
    phi3_analyzer = Phi3Analyzer()
    
    # 使用摘要进行分析（如果有的话）
    text_to_analyze = ""
    if 'abstract' in paper and pd.notna(paper['abstract']):
        text_to_analyze = paper['abstract']
        print(f"\n📝 原始摘要: {text_to_analyze}")
    
    # 如果有 URL，尝试下载 PDF
    if 'url' in paper and pd.notna(paper['url']):
        url = paper['url']
        print(f"\n🔗 论文链接: {url}")
        
        # 如果 URL 看起来像 PDF，尝试下载
        if 'pdf' in url.lower():
            print("📥 尝试下载 PDF...")
            pdf_path = pdf_processor.download_pdf(url, paper['title'])
            if pdf_path:
                print(f"✅ PDF 下载成功: {pdf_path}")
                
                # 提取 PDF 文本
                extracted_text = pdf_processor.extract_text_from_pdf(pdf_path)
                if extracted_text:
                    processed_text = pdf_processor.preprocess_text(extracted_text)
                    text_to_analyze = processed_text
                    print(f"📄 PDF 文本提取成功，长度: {len(processed_text)} 字符")
            else:
                print("❌ PDF 下载失败")
    
    # 进行 AI 分析
    if text_to_analyze:
        print("\n🤖 生成 AI 分析...")
        
        # 生成摘要
        ai_summary = phi3_analyzer.generate_summary(text_to_analyze)
        print(f"\n📋 AI 摘要:\n{ai_summary}")
        
        # 提取关键词
        keywords = phi3_analyzer.extract_keywords(text_to_analyze)
        print(f"\n🏷️ 关键词: {', '.join(keywords)}")
    else:
        print("\n❌ 没有可分析的文本内容")

def example_batch_processing():
    """批量处理示例"""
    print("\n\n⚡ 批量处理示例")
    print("=" * 50)
    
    csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 只处理前 3 篇论文作为示例
    sample_papers = df.head(3)
    
    print(f"📊 批量处理 {len(sample_papers)} 篇论文...")
    
    # 初始化分析器
    phi3_analyzer = Phi3Analyzer()
    
    results = []
    
    for i, (idx, paper) in enumerate(sample_papers.iterrows()):
        print(f"\n📄 处理论文 {i+1}/{len(sample_papers)}: {paper['title'][:50]}...")
        
        # 使用摘要进行分析
        text_to_analyze = ""
        if 'abstract' in paper and pd.notna(paper['abstract']):
            text_to_analyze = paper['abstract']
        
        if text_to_analyze:
            # 生成摘要
            ai_summary = phi3_analyzer.generate_summary(text_to_analyze)
            
            # 提取关键词
            keywords = phi3_analyzer.extract_keywords(text_to_analyze)
            
            result = {
                'title': paper['title'],
                'original_abstract': text_to_analyze,
                'ai_summary': ai_summary,
                'keywords': ', '.join(keywords)
            }
            results.append(result)
            
            print(f"  ✅ 完成")
        else:
            print(f"  ❌ 跳过（无摘要）")
    
    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        output_file = "output/batch_analysis_results.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 结果已保存到: {output_file}")
        
        # 显示结果摘要
        print("\n📋 处理结果摘要:")
        for i, result in enumerate(results):
            print(f"\n论文 {i+1}: {result['title'][:60]}...")
            print(f"  AI摘要: {result['ai_summary'][:100]}...")
            print(f"  关键词: {result['keywords']}")

def main():
    """主函数 - 运行所有示例"""
    print("🚀 Phi-3 论文分析系统 - 使用示例")
    print("=" * 60)
    
    try:
        # 检查必要的目录
        Path("log").mkdir(exist_ok=True)
        Path("example_pdfs").mkdir(exist_ok=True)
        Path("output").mkdir(exist_ok=True)
        
        # 运行示例
        example_basic_usage()
        example_csv_processing()
        example_single_paper_analysis()
        example_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("\n💡 接下来您可以：")
        print("  1. 修改参数来适应您的需求")
        print("  2. 添加更多的分析功能")
        print("  3. 集成到您的工作流程中")
        
    except Exception as e:
        print(f"\n❌ 运行示例时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
