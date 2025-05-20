import argparse
import datetime
import json
import logging
import os
import random
import time
import re
from typing import List, Dict, Optional, Tuple
import requests

import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log/snowballing_spider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class SnowballingSpider:
    """ 实现基于 Google Scholar 的向前和向后滚雪球功能。 """

    def __init__(self,
                 max_results_per_direction: int = 50,
                 delay_range: tuple = (3, 6),
                 max_retries: int = 3,
                 timeout: int = 40):
        """
        初始化滚雪球爬虫。

        Args:
            max_results_per_direction: 每个方向（向前/向后）滚雪球时获取的最大文献数量。
            delay_range: 随机延迟范围（秒）。
            max_retries: 网络请求或页面元素查找的最大重试次数。
            timeout: 页面加载和元素查找的超时时间（秒）。
        """
        self.max_results_per_direction = max_results_per_direction
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.timeout = timeout
        self.driver = None
        self.ua = UserAgent()

    def _setup_driver(self) -> None:
        """设置并初始化 Chrome 驱动。"""
        if self.driver:
            return
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument(f"user-agent={self.ua.random}")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.set_script_timeout(self.timeout)
            logging.info("Chrome 驱动初始化成功。")
        except Exception as e:
            logging.error(f"Chrome 驱动初始化失败: {str(e)}")
            raise

    def _get_page_content(self, url: str, wait_for_class: Optional[str] = "gs_r") -> Optional[str]:
        """获取指定URL的页面内容，并等待特定类名的元素出现。""" 
        self._setup_driver()
        for attempt in range(self.max_retries):
            try:
                self.driver.get(url)
                if wait_for_class:
                    WebDriverWait(self.driver, self.timeout).until(
                        EC.presence_of_element_located((By.CLASS_NAME, wait_for_class))
                    )
                # 尝试滚动页面以加载更多内容或触发懒加载
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(0.5, 1.5)) # 等待滚动和可能的动态加载
                return self.driver.page_source
            except TimeoutException:
                logging.warning(f"页面加载超时 ({url})，尝试重试 ({attempt + 1}/{self.max_retries})")
            except Exception as e:
                logging.error(f"获取页面内容时出错 ({url}): {str(e)}")
            
            if attempt < self.max_retries - 1:
                time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))
            else:
                logging.error(f"多次尝试后无法获取页面内容: {url}")
                return None
        return None

    def _parse_single_article_entry(self, div: BeautifulSoup) -> Optional[Dict]:
        """解析单个文献条目HTML元素，提取信息。与1中的解析逻辑类似。"""
        try:
            title_tag = div.find("h3", class_="gs_rt")
            if not title_tag: return None
            title = title_tag.get_text(strip=True)
            url = title_tag.find("a")["href"] if title_tag.find("a") else None

            author_info_div = div.find("div", class_="gs_a")
            first_author, year, journal = None, None, None
            if author_info_div:
                author_text = author_info_div.get_text(strip=True)
                authors_part = author_text.split(" - ")[0]
                if authors_part:
                    first_author = authors_part.split(",")[0].strip()
                year_match = re.search(r'\\b(?:19|20)\\d{2}\\b', author_text)
                year = int(year_match.group(0)) if year_match else None
                journal_parts = author_text.split(" - ")
                if len(journal_parts) > 1: # simplistic journal extraction
                    journal = journal_parts[1].split(",")[0].strip()


            abstract_div = div.find("div", class_="gs_rs")
            abstract = abstract_div.get_text(strip=True) if abstract_div else None

            citations = 0
            gs_fl_div = div.find("div", class_="gs_flb") or div.find("div", class_="gs_fl")
            if gs_fl_div:
                citation_link = gs_fl_div.find("a", href=re.compile(r'/scholar\?cites=\d+'))
                if citation_link:
                    citation_text = citation_link.get_text(strip=True)
                    citations_match = re.search(r'\d+', citation_text)
                    if citations_match:
                        citations = int(citations_match.group(0))
            
            return {
                "title": title, "url": url, "authors": first_author,
                "year": year, "journal": journal, "abstract": abstract,
                "citations": citations
            }
        except Exception as e:
            logging.warning(f"解析单个文献条目时出错: {title_tag.get_text(strip=True) if title_tag else '未知标题'} - {str(e)}")
            return None

    def _scrape_articles_from_page_content(self, page_content: str) -> List[Dict]:
        """从给定的HTML内容中抓取所有文献条目。"""
        if not page_content:
            return []
        
        soup = BeautifulSoup(page_content, 'html.parser')
        articles = []
        for div in soup.find_all("div", class_="gs_r"):
            article_data = self._parse_single_article_entry(div)
            if article_data:
                articles.append(article_data)
        return articles

    def _find_target_article_link(self, seed_title_query: str, link_text_pattern: str) -> Optional[str]:
        """
        在Google Scholar搜索结果中找到与种子标题最匹配的文章，并返回其特定操作链接的URL。
        例如，link_text_pattern可以是 "被引用" 或 "相关文章"。
        """
        self._setup_driver()
        search_url = f"https://scholar.google.com/scholar?hl=en&q={requests.utils.quote(seed_title_query)}"
        logging.info(f"正在搜索种子文章: {seed_title_query} (URL: {search_url})")
        
        page_content = self._get_page_content(search_url)
        if not page_content:
            logging.error(f"无法获取种子文章 '{seed_title_query}' 的搜索结果页面。")
            return None

        soup = BeautifulSoup(page_content, 'html.parser')
        
        # 尝试找到最相关的文章条目 (通常是第一个，或标题最相似的)
        # 这里的匹配逻辑可以更复杂，例如使用模糊匹配，但现在我们先用简单的方式
        target_article_div = None
        search_results = soup.find_all("div", class_="gs_r")
        if not search_results:
            logging.warning(f"未找到关于 '{seed_title_query}' 的搜索结果。")
            return None

        # 简单地取第一个结果作为目标文章的容器，或者进行更精确的匹配
        # 此处我们先假设第一个结果是目标文章
        target_article_div = search_results[0] 
        
        actual_title_tag = target_article_div.find("h3", class_="gs_rt")
        if actual_title_tag:
            logging.info(f"假定目标文章为: {actual_title_tag.get_text(strip=True)}")
        else:
            logging.warning("在第一个搜索结果中未找到标题。")
            return None # 或者尝试下一个结果

        # 在目标文章条目中查找包含特定文本模式的链接
        footer_div = target_article_div.find("div", class_="gs_flb") or target_article_div.find("div", class_="gs_fl")
        if not footer_div:
            logging.warning(f"未找到文章 '{seed_title_query}' 的底部链接区域。")
            return None

        target_link_tag = footer_div.find("a", text=re.compile(link_text_pattern, re.IGNORECASE))
        if target_link_tag and target_link_tag.get("href"):
            action_url = "https://scholar.google.com" + target_link_tag["href"]
            logging.info(f"找到操作链接 '{link_text_pattern}' for '{seed_title_query}': {action_url}")
            return action_url
        else:
            # 特殊处理 "被引用次数：XXX" 这种链接，因为它的文本不完全是 "被引用"
            if "被引用" in link_text_pattern or "Cited by" in link_text_pattern:
                 cited_by_link = footer_div.find("a", href=re.compile(r'/scholar\?cites=\d+'))
                 if cited_by_link and cited_by_link.get("href"):
                    action_url = "https://scholar.google.com" + cited_by_link["href"]
                    logging.info(f"通过href找到引用链接 for '{seed_title_query}': {action_url}")
                    return action_url
            logging.warning(f"未找到文本模式为 '{link_text_pattern}' 的链接 for '{seed_title_query}'。")
            return None

    def _fetch_all_articles_from_action_url(self, action_url: str, direction: str) -> List[Dict]:
        """从给定的操作URL（如"被引用"页面）开始，抓取所有文献直到达到max_results。"""
        all_articles = []
        current_url = action_url
        
        with tqdm(total=self.max_results_per_direction, desc=f"{direction}滚雪球") as pbar:
            while current_url and len(all_articles) < self.max_results_per_direction:
                logging.info(f"正在处理 {direction} 页面: {current_url}")
                page_content = self._get_page_content(current_url)
                if not page_content:
                    break # 无法获取页面，终止

                articles_on_page = self._scrape_articles_from_page_content(page_content)
                if not articles_on_page:
                    logging.info(f"页面 {current_url} 未找到文献，或已到达末尾。")
                    break 

                for article in articles_on_page:
                    if len(all_articles) < self.max_results_per_direction:
                        all_articles.append(article)
                        pbar.update(1)
                    else:
                        break
                
                if len(all_articles) >= self.max_results_per_direction:
                    break

                # 查找下一页链接
                soup = BeautifulSoup(page_content, 'html.parser')
                current_url_params = {} 
                if current_url and '?' in current_url:
                    try:
                        current_url_params = dict(item.split("=") for item in current_url.split('?')[1].split("&"))
                    except ValueError:
                        logging.warning(f"解析当前URL参数时出错: {current_url}")
                
                current_start = int(current_url_params.get("start", 0))
                next_start = current_start + 10
                
                # 尝试找到 href 中包含 start=next_start 的链接
                # 这通常是数字页码链接，如 <a href="...start=10...">2</a>
                next_page_tag = soup.find("a", href=re.compile(f"start={next_start}"))
                
                # 备选方案：查找文本为下一页页码的链接 (更脆弱)
                if not next_page_tag:
                    next_page_number_text = str((next_start // 10) + 1)
                    # 查找 class="gs_nma" 且文本是下一页页码的链接
                    page_links_container = soup.find("div", id="gs_nml_pg") # 通常页码链接在这个div里
                    if page_links_container:
                        next_page_tag = page_links_container.find("a", class_="gs_nma", text=next_page_number_text)
                    if not next_page_tag and page_links_container: # 如果上面没找到，再尝试不带 class 的页码链接
                        next_page_tag = page_links_container.find("a", text=next_page_number_text)

                next_page_href = None
                if next_page_tag:
                    href_value = next_page_tag.get("href")
                    if href_value and not href_value.lower().startswith("javascript:"):
                        # 确保 href 是相对路径或包含 scholar.google.com
                        if href_value.startswith("/") or "scholar.google.com" in href_value:
                            next_page_href = href_value
                        else:
                            logging.warning(f"找到的下一页href格式不符合预期: {href_value}")
                    
                if next_page_href:
                    if next_page_href.startswith("/"):
                        current_url = "https://scholar.google.com" + next_page_href
                    else:
                        current_url = next_page_href # 假定是完整URL
                    time.sleep(random.uniform(self.delay_range[0] / 2, self.delay_range[1] / 2))
                else:
                    logging.info(f"在引用/相关文章页面未找到有效的下一页链接 (尝试start={next_start})，{direction}滚雪球结束。")
                    current_url = None
            
        return all_articles[:self.max_results_per_direction]

    def save_to_csv(self, articles: List[Dict], filename: str) -> None:
        """将文献列表保存到CSV文件。"""
        if not articles:
            logging.info(f"没有文献数据可保存到 {filename}。")
            return
        try:
            df = pd.DataFrame(articles)
            # 确保列的顺序一致性，可以定义一个列的列表
            columns_order = ["title", "authors", "year", "journal", "citations", "abstract", "url"]
            df = df.reindex(columns=[col for col in columns_order if col in df.columns])
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logging.info(f"结果已成功保存到: {filename} (共 {len(articles)} 条记录)")
        except Exception as e:
            logging.error(f"保存结果到 {filename} 时出错: {str(e)}")

    def perform_snowballing(self, seed_article_title: str):
        """
        对给定的种子文章标题执行向前和向后滚雪球。
        """
        self._setup_driver()
        
        # --- 向后滚雪球 (Cited By) ---
        logging.info(f"--- 开始向后滚雪球 for '{seed_article_title}' ---")
        # "被引用" or "Cited by"
        cited_by_action_url = self._find_target_article_link(seed_article_title, r"(被引用|Cited by)")
        backward_articles = []
        if cited_by_action_url:
            backward_articles = self._fetch_all_articles_from_action_url(cited_by_action_url, "向后")
        else:
            logging.warning(f"未能找到 '{seed_article_title}' 的引用链接，无法进行向后滚雪球。")
        
        self.save_to_csv(backward_articles, f"output/backward_snowball_results_{seed_article_title[:20].replace(' ', '_')}.csv")
        # --- 向前滚雪球 (Related Articles) ---
        logging.info(f"--- 开始向前滚雪球 for '{seed_article_title}' ---")
        # "相关文章" or "Related articles"
        related_articles_action_url = self._find_target_article_link(seed_article_title, r"(相关文章|Related articles)")
        forward_articles = []
        if related_articles_action_url:
            forward_articles = self._fetch_all_articles_from_action_url(related_articles_action_url, "向前")
        else:
            logging.warning(f"未能找到 '{seed_article_title}' 的相关文章链接，无法进行向前滚雪球。")

        self.save_to_csv(forward_articles, f"output/forward_snowball_results_{seed_article_title[:20].replace(' ', '_')}.csv")
        if self.driver:
            self.driver.quit()
            self.driver = None
            logging.info("Chrome 驱动已关闭。")
        
        logging.info("滚雪球操作完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于Google Scholar的向前和向后滚雪球爬虫。")
    parser.add_argument("seed_title", type=str, help="用于开始滚雪球的种子文章的确切标题。")
    parser.add_argument("--max_results", type=int, default=50, help="每个滚雪球方向获取的最大文献数。")
    parser.add_argument("--timeout", type=int, default=40, help="页面加载和元素查找的超时时间（秒）。")

    args = parser.parse_args()

    if not args.seed_title:
        logging.error("错误：必须提供种子文章标题。使用 --help 查看更多信息。")
    else:
        try:
            spider = SnowballingSpider(
                max_results_per_direction=args.max_results,
                timeout=args.timeout
            )
            spider.perform_snowballing(args.seed_title)
        except Exception as e:
            logging.error(f"执行滚雪球操作时发生严重错误: {str(e)}")
            traceback.print_exc()

# 示例用法:
# python my_snow_balling.py "Data-driven design by analogy: state-of-the-art and future directions" --max_results 30
