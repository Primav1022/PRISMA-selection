#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 功能测试脚本
==============

测试 PyTorch GPU 功能是否正常工作
"""

import torch
import time

def test_gpu_basic():
    """基础 GPU 测试"""
    print("🔍 基础 GPU 测试")
    print("-" * 40)
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"当前 GPU: {torch.cuda.current_device()}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        
        # 显示 GPU 内存信息
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 内存: {gpu_memory:.2f} GB")
        
        # 显示当前内存使用情况
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"已分配内存: {allocated:.2f} GB")
        print(f"缓存内存: {cached:.2f} GB")
        
        return True
    else:
        print("❌ CUDA 不可用")
        return False

def test_gpu_computation():
    """GPU 计算测试"""
    print("\n🧮 GPU 计算测试")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过计算测试")
        return
    
    # 创建测试数据
    size = 1000
    print(f"创建 {size}x{size} 矩阵进行测试...")
    
    # CPU 计算
    print("⏱️  CPU 计算测试...")
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU 计算时间: {cpu_time:.4f} 秒")
    
    # GPU 计算
    print("🚀 GPU 计算测试...")
    device = torch.device('cuda')
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    
    # 预热 GPU
    torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start_time = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU 计算时间: {gpu_time:.4f} 秒")
    
    # 计算加速比
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"🏃‍♂️ GPU 加速比: {speedup:.2f}x")
    
    # 验证结果正确性
    c_gpu_cpu = c_gpu.cpu()
    max_diff = torch.max(torch.abs(c_cpu - c_gpu_cpu)).item()
    print(f"✅ 计算结果差异: {max_diff:.2e} (应该接近0)")

def test_transformers_gpu():
    """测试 transformers 库的 GPU 支持"""
    print("\n🤖 Transformers GPU 测试")
    print("-" * 40)
    
    try:
        from transformers import AutoTokenizer
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA 不可用，跳过 transformers GPU 测试")
            return
        
        print("✅ transformers 库导入成功")
        print("✅ GPU 设备可用")
        print("🎯 建议使用较小的模型进行测试，如 'gpt2' 或 'distilbert-base-uncased'")
        
        # 测试一个小模型（如果用户想要的话）
        test_small_model = input("\n是否测试小模型 GPT-2？(y/n): ").lower().strip()
        if test_small_model == 'y':
            print("📥 加载 GPT-2 模型...")
            from transformers import AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            model = AutoModelForCausalLM.from_pretrained('gpt2')
            
            # 移动到 GPU
            device = torch.device('cuda')
            model = model.to(device)
            
            print("✅ 模型已加载到 GPU")
            print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 简单测试
            text = "The future of AI is"
            inputs = tokenizer(text, return_tensors='pt').to(device)
            
            print(f"🔤 测试文本: '{text}'")
            print("🎯 生成中...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20, 
                    do_sample=True, 
                    temperature=0.7
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"✨ 生成结果: '{generated_text}'")
            print("✅ GPU 模型测试成功！")
        
    except ImportError:
        print("❌ transformers 库未安装")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    """主函数"""
    print("🚀 GPU 功能测试开始")
    print("=" * 50)
    
    # 基础 GPU 测试
    gpu_available = test_gpu_basic()
    
    if gpu_available:
        # 计算性能测试
        test_gpu_computation()
        
        # Transformers GPU 测试
        test_transformers_gpu()
    
    print("\n" + "=" * 50)
    print("🏁 测试完成！")
    
    if gpu_available:
        print("✅ 您的 GPU 设置正常，可以运行 Phi-3 分析系统")
        print("💡 如果模型下载遇到问题，可以：")
        print("   1. 使用 HuggingFace 镜像")
        print("   2. 手动下载模型文件")
        print("   3. 使用较小的模型进行测试")
    else:
        print("⚠️  GPU 不可用，系统将使用 CPU 模式（速度较慢）")

if __name__ == "__main__":
    main()
