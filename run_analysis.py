#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文分析系统启动脚本
==================

这是一个简化的启动脚本，帮助用户快速开始使用 Phi-3 论文分析系统。

使用方法:
    python run_analysis.py
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy', 
        'requests', 'tqdm', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def setup_directories():
    """创建必要的目录"""
    dirs = ['log', 'pdfs', 'output']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ 目录设置完成")

def import_analysis_module():
    """导入分析模块"""
    try:
        spec = importlib.util.spec_from_file_location(
            "analysis_module", 
            "5_intelligent_with_phi3.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ 无法导入分析模块: {e}")
        return None

def run_simple_demo():
    """运行简单演示"""
    print("\n🔬 运行简单演示...")
    
    # 导入分析模块
    analysis_module = import_analysis_module()
    if not analysis_module:
        return
    
    try:
        # 检查是否有数据文件
        csv_file = "output/merged_cleaned_results_intelligent_cleaned.csv"
        if not os.path.exists(csv_file):
            print(f"❌ 找不到数据文件: {csv_file}")
            print("请先运行前面的数据收集步骤 (1-4)")
            return
        
        # 初始化分析器（仅用于测试）
        print("📝 初始化 Phi-3 分析器...")
        analyzer = analysis_module.Phi3Analyzer()
        
        # 测试文本
        test_text = """
        This research presents a comprehensive study on artificial intelligence 
        applications in healthcare. The methodology combines machine learning 
        algorithms with clinical data to improve diagnostic accuracy. Results 
        show significant improvements in patient outcomes and reduced costs.
        """
        
        print("🤖 生成测试摘要...")
        summary = analyzer.generate_summary(test_text)
        print(f"摘要: {summary}")
        
        print("🏷️ 提取关键词...")
        keywords = analyzer.extract_keywords(test_text)
        print(f"关键词: {', '.join(keywords)}")
        
        print("\n✅ 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        import traceback
        traceback.print_exc()

def show_menu():
    """显示菜单选项"""
    print("\n" + "="*50)
    print("📚 Phi-3 论文分析系统")
    print("="*50)
    print("请选择操作:")
    print("1. 运行简单演示")
    print("2. 查看使用指南")
    print("3. 检查系统状态")
    print("4. 退出")
    print("-"*50)

def show_usage_guide():
    """显示使用指南"""
    print("\n📖 使用指南:")
    print("1. 确保已安装所有依赖: pip install -r requirements.txt")
    print("2. 准备论文数据 CSV 文件")
    print("3. 运行分析脚本: python 5_intelligent_with_phi3.py")
    print("4. 查看结果文件")
    print("\n详细文档请查看: USAGE_GUIDE.md")

def check_system_status():
    """检查系统状态"""
    print("\n🔍 系统状态检查:")
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查依赖包
    if check_dependencies():
        print("✅ 所有依赖包已安装")
    
    # 检查文件
    files_to_check = [
        "5_intelligent_with_phi3.py",
        "requirements.txt", 
        "USAGE_GUIDE.md"
    ]
    
    for file_name in files_to_check:
        if os.path.exists(file_name):
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 缺失")
    
    # 检查目录
    dirs_to_check = ['log', 'pdfs', 'output']
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ 目录存在")
        else:
            print(f"❌ {dir_name}/ 目录缺失")

def main():
    """主函数"""
    print("🚀 启动论文分析系统...")
    
    # 设置目录
    setup_directories()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选项 (1-4): ").strip()
            
            if choice == '1':
                if check_dependencies():
                    run_simple_demo()
                else:
                    print("请先安装依赖包")
            
            elif choice == '2':
                show_usage_guide()
            
            elif choice == '3':
                check_system_status()
            
            elif choice == '4':
                print("👋 再见！")
                break
            
            else:
                print("❌ 无效选项，请重新选择")
        
        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
