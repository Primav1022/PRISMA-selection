#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化系统测试 - 使用小模型测试分析功能
=====================================

这个脚本使用较小的模型来测试论文分析系统的基本功能，
避免下载大型 Phi-3 模型的网络问题。
"""

import os
import sys
import pandas as pd
import torch
from pathlib import Path

# 设置环境变量使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def test_basic_analysis():
    """使用小模型测试基本分析功能"""
    print("🔬 基本分析功能测试")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 使用较小的中文模型进行测试
        model_name = "uer/gpt2-chinese-cluecorpussmall"
        print(f"📥 加载模型: {model_name}")
        
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 移动到 GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"✅ 模型加载成功，使用设备: {device}")
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试文本生成
        test_text = "人工智能在学术研究中的应用"
        print(f"🔤 测试文本: '{test_text}'")
        
        inputs = tokenizer(test_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✨ 生成结果: '{generated_text}'")
        print("✅ 文本生成测试成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_pdf_processing():
    """测试 PDF 处理功能"""
    print("\n📄 PDF 处理功能测试")
    print("-" * 40)
    
    try:
        # 导入 PDF 处理模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("analysis", "5_intelligent_with_phi3.py")
        analysis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis_module)
        
        # 创建 PDF 处理器
        pdf_processor = analysis_module.PDFProcessor()
        print("✅ PDF 处理器创建成功")
        
        # 测试文本预处理
        test_text = """
        This is a test document with multiple    spaces
        and
        
        multiple
        line breaks.
        
        It should be cleaned up by the preprocessing function.
        """
        
        processed_text = pdf_processor.preprocess_text(test_text)
        print("🔧 文本预处理测试:")
        print(f"原始长度: {len(test_text)}")
        print(f"处理后长度: {len(processed_text)}")
        print(f"处理结果: {processed_text[:100]}...")
        print("✅ 文本预处理测试成功！")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF 处理测试失败: {e}")
        return False

def test_csv_reading():
    """测试 CSV 文件读取"""
    print("\n📊 CSV 文件读取测试")
    print("-" * 40)
    
    csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到文件: {csv_file}")
        print("请确保已运行前面的数据收集步骤")
        return False
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"✅ 成功读取 CSV 文件")
        print(f"📈 论文数量: {len(df)}")
        print(f"📋 列名: {list(df.columns)}")
        
        # 显示前几条记录的基本信息
        print("\n📝 前 3 条记录:")
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
            print(f"\n  论文 {i+1}:")
            print(f"    标题: {row['title'][:60]}...")
            print(f"    作者: {row['authors']}")
            print(f"    年份: {row['year']}")
            print(f"    引用数: {row['citations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV 读取失败: {e}")
        return False

def create_simple_demo():
    """创建简单演示"""
    print("\n🎯 创建简单演示")
    print("-" * 40)
    
    # 创建示例数据
    sample_data = {
        'title': [
            'Artificial Intelligence in Healthcare: A Comprehensive Review',
            'Machine Learning Applications in Academic Research',
            'Deep Learning for Natural Language Processing'
        ],
        'authors': [
            'Zhang, L.; Wang, M.; Li, H.',
            'Smith, J.; Brown, A.',
            'Johnson, K.; Davis, R.'
        ],
        'year': [2023, 2022, 2024],
        'citations': [45, 32, 18],
        'abstract': [
            'This paper presents a comprehensive review of artificial intelligence applications in healthcare, focusing on machine learning algorithms and their clinical implementations.',
            'We explore various machine learning techniques used in academic research, including data mining, predictive modeling, and statistical analysis methods.',
            'This study investigates deep learning approaches for natural language processing tasks, with emphasis on transformer architectures and attention mechanisms.'
        ]
    }
    
    # 保存为 CSV
    df = pd.DataFrame(sample_data)
    demo_file = "output/demo_papers.csv"
    df.to_csv(demo_file, index=False, encoding='utf-8-sig')
    print(f"✅ 创建演示数据: {demo_file}")
    
    return demo_file

def main():
    """主函数"""
    print("🚀 简化系统测试开始")
    print("=" * 50)
    
    # 确保目录存在
    Path("output").mkdir(exist_ok=True)
    Path("log").mkdir(exist_ok=True)
    
    results = []
    
    # 测试基本分析功能
    results.append(("GPU/模型测试", test_basic_analysis()))
    
    # 测试 PDF 处理
    results.append(("PDF处理测试", test_pdf_processing()))
    
    # 测试 CSV 读取
    results.append(("CSV读取测试", test_csv_reading()))
    
    # 创建演示数据
    demo_file = create_simple_demo()
    results.append(("演示数据创建", demo_file is not None))
    
    # 显示测试结果
    print("\n" + "=" * 50)
    print("📋 测试结果总结:")
    print("-" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\n📊 总体结果: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！系统基本功能正常")
        print("\n💡 下一步建议:")
        print("1. 尝试使用较小的中文模型替代 Phi-3")
        print("2. 或者手动下载 Phi-3 模型到本地")
        print("3. 使用演示数据测试完整流程")
    else:
        print("⚠️  部分测试失败，请检查相关配置")
    
    print("\n🔧 如果要使用 Phi-3 模型，建议:")
    print("1. 配置网络代理或使用镜像站点")
    print("2. 手动下载模型文件")
    print("3. 或使用其他兼容的中文大语言模型")

if __name__ == "__main__":
    main()
