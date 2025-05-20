# 📚 PRISMA Selection Code

这是一个用于自动化文献检索和筛选的工具集，基于Google Scholar进行论文爬取、滚雪球检索、查重和智能分析。

## 🛠️ 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保已安装Chrome浏览器（用于Selenium爬虫）

## 📋 功能模块

### 1. 📚 论文检索 (1_scholar_crawler.py)
从Google Scholar爬取论文信息
```bash
python scholar_crawler.py "your search query" --max_results 100
```
- 支持关键词搜索
- 可设置最大结果数
- 自动保存为CSV文件

### 2. 🌨️ 滚雪球检索 (snowballing.py)
基于种子论文进行前向和后向引用检索
```bash
python snowballing.py "paper title" --max_results 30
```
- 支持前向引用（被谁引用）
- 支持后向引用（引用了谁）
- 自动去重和合并结果

### 3. 🔄 查重处理 (duplicate_check.py)
对检索结果进行查重和清理
```bash
python duplicate_check.py
```
- 基于标题相似度
- 自动合并重复条目
- 生成清理后的CSV文件

### 4. 🧠 智能分析 (intelligent_check.py)
对合并后的结果进行智能分析
```bash
python intelligent_check.py
```
- 基于内容相似度
- 自动分类和标记
- 生成分析报告

## 📁 文件结构
├── output/ # 输出文件目录
│ ├── .csv # 生成的CSV文件
│ └── merged_results/ # 合并后的结果
├── log/ # 日志文件目录
├── scholar_crawler.py # 论文检索模块
├── snowballing.py # 滚雪球检索模块
├── duplicate_check.py # 查重处理模块
├── intelligent_check.py # 智能分析模块



## ⚠️ 使用注意事项

1. **爬虫限制**：
   - 建议设置适当的延迟（默认已配置）
   - 避免频繁请求以防IP被封

2. **文件处理**：
   - 所有输出文件保存在 `output` 目录
   - 日志文件保存在 `log` 目录

3. **内存使用**：
   - 处理大量数据时注意内存占用
   - 建议分批处理大型数据集

## 🔄 工作流程

1. 使用 `1_scholar_crawler.py` 获取初始论文列表
2. 使用 `2_snowballing.py` 扩展检索范围
3. 使用 `3_duplicate_check.py` 清理重复数据
4. 使用 `4_intelligent_check.py` 进行智能分析


## 🐛 常见问题

1. **Chrome驱动问题**：
   - 确保Chrome浏览器版本与驱动匹配
   - 使用 `webdriver-manager` 自动管理驱动

2. **编码问题**：
   - 所有文件使用 UTF-8 编码
   - 确保系统支持中文显示

3. **目前案例**：
   - 都是可以删除的，作者自己systematic review的残余~~

