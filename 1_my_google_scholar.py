import argparse
import datetime
import json
import logging
import os
import random
import time
from typing import List, Dict, Optional, Union
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import re

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/scholar_spider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class GoogleScholarSpider:
    """Google Scholar 爬虫类"""
    
    def __init__(self, 
                 max_results: int = 100,
                 delay_range: tuple = (2, 5),
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        初始化爬虫
        
        Args:
            max_results: 最大结果数
            delay_range: 随机延迟范围（秒）
            max_retries: 最大重试次数
            timeout: 页面加载超时时间（秒）
        """
        self.max_results = max_results
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.timeout = timeout
        self.driver = None
        self.current_year = datetime.datetime.now().year
        
    def setup_driver(self) -> None:
        """设置并初始化 Chrome 驱动"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # 无头模式
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument(f"user-agent={UserAgent().random}")
            
            # 添加实验性选项
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # 使用 ChromeDriverManager 自动匹配版本
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # 设置页面加载超时
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.set_script_timeout(self.timeout)
            
            logging.info("Chrome 驱动初始化成功")
            
        except Exception as e:
            logging.error(f"Chrome 驱动初始化失败: {str(e)}")
            raise
    
    def search(self, 
               query: str,
               author: Optional[str] = None,
               year_start: Optional[int] = None,
               year_end: Optional[int] = None,
               sort_by: str = "relevance") -> List[Dict]:
        """
        搜索 Google Scholar
        
        Args:
            query: 搜索关键词
            author: 作者名
            year_start: 起始年份
            year_end: 结束年份
            sort_by: 排序方式 ("relevance" 或 "date")
            
        Returns:
            搜索结果列表
        """
        if not self.driver:
            self.setup_driver()
            
        results = []
        page = 0
        
        try:
            with tqdm(total=self.max_results, desc="获取搜索结果") as pbar:
                while len(results) < self.max_results:
                    # 构建搜索 URL
                    url = self._build_search_url(query, author, year_start, year_end, sort_by, page)
                    logging.info(f"正在获取第 {page + 1} 页结果")
                    
                    # 获取页面内容
                    content = self._get_page_content(url)
                    if not content:
                        break
                        
                    # 解析结果
                    new_results = self._parse_results(content)
                    if not new_results:
                        break
                        
                    results.extend(new_results)
                    pbar.update(len(new_results))
                    
                    # 随机延迟
                    time.sleep(random.uniform(*self.delay_range))
                    page += 1
                    
            return results[:self.max_results]
            
        except Exception as e:
            logging.error(f"搜索过程中出错: {str(e)}")
            return results
            
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def _build_search_url(self, 
                         query: str,
                         author: Optional[str],
                         year_start: Optional[int],
                         year_end: Optional[int],
                         sort_by: str,
                         page: int) -> str:
        """构建搜索 URL"""
        base_url = "https://scholar.google.com/scholar"
        params = {
            "q": query,
            "start": page * 10,
            "hl": "en",
            "as_sdt": "0,5"
        }
        
        if author:
            params["as_sauthors"] = author
            
        if year_start:
            params["as_ylo"] = year_start
            
        if year_end:
            params["as_yhi"] = year_end
            
        if sort_by == "date":
            params["scisbd"] = "1"
            
        return f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    
    def _get_page_content(self, url: str) -> Optional[str]:
        """获取页面内容"""
        for attempt in range(self.max_retries):
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, self.timeout).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "gs_r"))
                )
                return self.driver.page_source
                
            except TimeoutException:
                logging.warning(f"页面加载超时，尝试重试 ({attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(random.uniform(*self.delay_range))
                
            except Exception as e:
                logging.error(f"获取页面内容时出错: {str(e)}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(random.uniform(*self.delay_range))
    
    def _parse_results(self, content: str) -> List[Dict]:
        """解析搜索结果"""
        results = []
        soup = BeautifulSoup(content, 'html.parser')
        
        for div in soup.find_all("div", class_="gs_r"):
            try:
                # 获取标题和链接
                title_tag = div.find("h3", class_="gs_rt")
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                url = title_tag.find("a")["href"] if title_tag.find("a") else None
                
                # 获取作者、年份和期刊信息
                author_info = div.find("div", class_="gs_a")
                if author_info:
                    author_text = author_info.get_text(strip=True)
                    # Extract authors part before potential ' - ' separator
                    authors_part = author_text.split(" - ")[0] if " - " in author_text else author_text
                    # Take only the first author if multiple are listed (usually comma-separated)
                    if authors_part:
                        first_author = authors_part.split(",")[0].strip()
                    else:
                        first_author = None
                    year = self._extract_year(author_text)
                    journal = author_text.split(" - ")[-1] if " - " in author_text else None
                else:
                    authors = year = journal = None # This should be first_author
                    first_author = year = journal = None # Corrected line
                
                # 获取摘要
                abstract = div.find("div", class_="gs_rs")
                abstract = abstract.get_text(strip=True) if abstract else None
                
                # 获取引用数
                citations = 0 # Initialize to 0
                
                # 优先查找包含 gs_flb 类的 div，如果找不到，则查找 gs_fl 类
                # div 是当前处理的单个搜索结果条目 (gs_r)
                gs_fl_div = div.find("div", class_="gs_flb") # 优先查找更具体的 class
                if not gs_fl_div: # 如果没找到 gs_flb, 退回查找 gs_fl
                    gs_fl_div = div.find("div", class_="gs_fl")

                if gs_fl_div:
                    # Strategy: Find the link whose href attribute indicates it's a citation link.
                    citation_link = gs_fl_div.find("a", href=re.compile(r'/scholar\?cites=\d+'))
                    
                    if citation_link:
                        # 添加调试日志，打印找到的链接的href和文本
                        logging.debug(f"找到引用链接 for '{title[:50]}...': href='{citation_link.get('href')}', text='{citation_link.get_text(strip=True)}'")
                        citation_text = citation_link.get_text(strip=True) # e.g., "Cited by 85", "被引用次数：85"
                        try:
                            citations_match = re.search(r'\d+', citation_text)
                            if citations_match:
                                citations = int(citations_match.group(0))
                            else:
                                # Log if link found but no number in text
                                logging.warning(f"找到引用链接但文本中无数字: '{citation_text}' for title '{title[:50]}...'")
                        except Exception as e:
                            logging.error(f"从引用文本 '{citation_text}' 提取数字时出错: {e}")
                    # else: # No link matching the href pattern was found
                        # logging.info(f"未找到基于href的引用链接. Title: {title[:30]}...")
                        # If the href search fails, citations will remain 0.
                
                results.append({
                    "title": title,
                    "url": url,
                    "authors": first_author, # Use first_author
                    "year": year,
                    "journal": journal,
                    "abstract": abstract,
                    "citations": citations
                })
                
            except Exception as e:
                logging.warning(f"解析结果时出错: {str(e)}")
                continue
                
        return results
    
    def _extract_year(self, text: str) -> Optional[int]:
        """从文本中提取年份"""
        import re
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        return int(years[0]) if years else None
    
    def save_results(self, 
                    results: List[Dict],
                    output_file: str,
                    format: str = "csv") -> None:
        """
        保存结果
        
        Args:
            results: 搜索结果列表
            output_file: 输出文件路径
            format: 输出格式 ("csv" 或 "json")
        """
        try:
            if format.lower() == "csv":
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False, encoding='utf-8-sig') # Changed encoding to utf-8-sig
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            logging.info(f"结果已保存到: {output_file}")
            
        except Exception as e:
            logging.error(f"保存结果时出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Google Scholar 爬虫")
    parser.add_argument("query", help="搜索关键词")
    parser.add_argument("--author", help="作者名")
    parser.add_argument("--year-start", type=int, help="起始年份")
    parser.add_argument("--year-end", type=int, help="结束年份")
    parser.add_argument("--max-results", type=int, default=100, help="最大结果数")
    parser.add_argument("--output", default="output/scholar_results", help="输出文件名（不含扩展名）")  #更改存储位置及名称
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="输出格式")
    
    args = parser.parse_args()
    
    try:
        spider = GoogleScholarSpider(max_results=args.max_results)
        results = spider.search(
            query=args.query,
            author=args.author,
            year_start=args.year_start,
            year_end=args.year_end
        )
        
        if results:
            output_file = f"{args.output}.{args.format}"
            spider.save_results(results, output_file, args.format)
        else:
            logging.warning("未找到任何结果")
            
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()
