# 📚 Phi-3 论文分析系统使用指南

## 🎯 功能概述

这个系统使用微软的 Phi-3-mini 模型来分析学术论文，提供以下功能：

- 📄 **PDF 文本提取**：自动下载和提取 PDF 文件中的文本
- 🤖 **智能摘要生成**：使用 AI 生成论文的中文摘要
- 🏷️ **关键词提取**：自动提取论文的关键词
- 📊 **批量处理**：支持批量分析多篇论文

## 🛠️ 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. PyTorch 安装（如果遇到问题）

如果在安装 PyTorch 时遇到 "另一个程序正在使用此文件" 的错误，请尝试：

```bash
# 方法 1: 使用 --user 参数
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --user

# 方法 2: 清理临时文件后重试
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 方法 3: 使用 conda（推荐）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. 创建必要目录

```bash
mkdir log pdfs output
```

## 🚀 快速开始

### 基础使用

```python
from intelligent_with_phi3 import PDFProcessor, Phi3Analyzer

# 初始化组件
pdf_processor = PDFProcessor()
phi3_analyzer = Phi3Analyzer()

# 分析文本
text = "Your research paper abstract or content here..."
summary = phi3_analyzer.generate_summary(text)
keywords = phi3_analyzer.extract_keywords(text)

print("摘要:", summary)
print("关键词:", keywords)
```

### 运行完整示例

```bash
python example_usage.py
```

## 📋 详细使用步骤

### 步骤 1: 准备数据

确保您有一个包含论文信息的 CSV 文件，文件应包含以下列：
- `title`: 论文标题
- `authors`: 作者
- `year`: 发表年份
- `abstract`: 摘要（可选）
- `url`: 论文链接（可选）

### 步骤 2: 运行分析

```python
# 导入必要模块
import pandas as pd
from intelligent_with_phi3 import PDFProcessor, Phi3Analyzer

# 读取数据
df = pd.read_csv("your_papers.csv")

# 初始化分析器
pdf_processor = PDFProcessor()
phi3_analyzer = Phi3Analyzer()

# 处理单篇论文
paper = df.iloc[0]  # 选择第一篇论文
text = paper['abstract']  # 使用摘要

# 生成分析结果
summary = phi3_analyzer.generate_summary(text)
keywords = phi3_analyzer.extract_keywords(text)
```

### 步骤 3: 批量处理

```python
results = []
for idx, paper in df.iterrows():
    if pd.notna(paper['abstract']):
        summary = phi3_analyzer.generate_summary(paper['abstract'])
        keywords = phi3_analyzer.extract_keywords(paper['abstract'])
        
        results.append({
            'title': paper['title'],
            'ai_summary': summary,
            'keywords': ', '.join(keywords)
        })

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("analysis_results.csv", index=False, encoding='utf-8-sig')
```

## ⚙️ 配置选项

### PDF 处理器配置

```python
pdf_processor = PDFProcessor(
    download_dir="my_pdfs"  # PDF 下载目录
)
```

### Phi-3 分析器配置

```python
phi3_analyzer = Phi3Analyzer(
    model_name="microsoft/Phi-3-mini-4k-instruct"  # 模型名称
)

# 生成摘要时的参数
summary = phi3_analyzer.generate_summary(
    text, 
    max_new_tokens=256  # 最大生成长度
)

# 提取关键词时的参数
keywords = phi3_analyzer.extract_keywords(
    text,
    max_new_tokens=100  # 最大生成长度
)
```

## 📊 输出格式

### 摘要输出示例
```
这篇论文研究了数据驱动的超材料设计方法。主要采用机器学习算法结合物理仿真来自动生成和优化材料结构。研究发现该方法能够显著提高材料性能，在航空航天、汽车和生物医学工程等领域具有重要应用价值。
```

### 关键词输出示例
```
['数据驱动设计', '超材料', '机器学习', '结构优化', '材料性能', '多尺度系统']
```

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案：检查网络连接，确保能访问 Hugging Face
   或使用镜像站点：export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **CUDA 内存不足**
   ```python
   # 使用 CPU 模式
   analyzer = Phi3Analyzer()
   # 模型会自动检测并使用 CPU
   ```

3. **PDF 下载失败**
   ```python
   # 检查 URL 是否有效
   # 某些网站可能需要特殊的请求头或认证
   ```

4. **文本编码问题**
   ```python
   # 确保使用正确的编码读取 CSV
   df = pd.read_csv("file.csv", encoding='utf-8')
   ```

### 性能优化建议

1. **GPU 加速**：如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch
2. **批量处理**：一次处理多个文本可以提高效率
3. **内存管理**：处理大量文件时注意内存使用

## 📁 文件结构

```
PRISMA/
├── 5_intelligent_with_phi3.py    # 主分析脚本
├── example_usage.py              # 使用示例
├── requirements.txt              # 依赖包列表
├── USAGE_GUIDE.md               # 使用指南（本文件）
├── log/                         # 日志文件
├── pdfs/                        # PDF 下载目录
└── output/                      # 输出结果
    ├── *.csv                    # 论文数据
    └── analysis_results.csv     # 分析结果
```

## 🤝 技术支持

如果遇到问题，请检查：
1. 所有依赖包是否正确安装
2. Python 版本是否兼容（推荐 3.8+）
3. 系统资源是否充足（特别是内存）

## 📈 扩展功能

您可以基于现有代码添加：
- 可视化功能（词云图、统计图表）
- 更多的文本分析功能
- 与其他 AI 模型的集成
- Web 界面
- 数据库存储

## 🔄 更新日志

- **v1.0**: 基础功能实现
  - PDF 文本提取
  - Phi-3 模型集成
  - 摘要生成和关键词提取
  - 批量处理支持
