"""
Web scraping module for collecting news articles.

Methodology:
- Use requests + BeautifulSoup4 for HTML parsing
- Parse RSS feeds to discover article URLs
- Extract title and body text from article pages
- Store results in structured CSV format
"""
import os
import time
import csv
import logging
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsScraper:
    """
    Web scraper for news articles.
    Demonstrates scraping methodology - actual requests depend on network access.
    """

    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (compatible; NewsClassifierBot/1.0)',
        'Accept': 'text/html,application/xhtml+xml',
    }

    def __init__(self, delay: float = 1.0, max_content_length: int = 2000):
        self.delay = delay  # Polite delay between requests
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def scrape_article(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape a single article page.
        Returns dict with 'title' and 'content' keys, or None on failure.
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title - try common patterns
            title = ''
            for selector in ['h1.article-title', 'h1.entry-title', 'h1', 'title']:
                el = soup.select_one(selector)
                if el:
                    title = el.get_text(strip=True)
                    break

            # Extract body - try common article containers
            content = ''
            for selector in ['article', 'div.article-body', 'div.entry-content', 'main', 'div.content']:
                el = soup.select_one(selector)
                if el:
                    paragraphs = el.find_all('p')
                    content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                    break
            if not content:
                content = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))

            time.sleep(self.delay)
            return {'title': title, 'content': content[:self.max_content_length], 'url': url}
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None

    def scrape_feed(self, rss_url: str) -> List[Dict[str, str]]:
        """
        Parse an RSS feed and return list of article metadata dicts.
        Each dict has 'title', 'link', 'description'.
        """
        articles = []
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'xml')
            for item in soup.find_all('item'):
                articles.append({
                    'title': item.find('title').get_text(strip=True) if item.find('title') else '',
                    'link': item.find('link').get_text(strip=True) if item.find('link') else '',
                    'description': item.find('description').get_text(strip=True) if item.find('description') else '',
                })
            logger.info(f"Found {len(articles)} articles in feed {rss_url}")
        except Exception as e:
            logger.warning(f"Failed to parse feed {rss_url}: {e}")
        return articles

    def collect_training_data(self, urls: List[str], category: str) -> List[Dict[str, str]]:
        """
        Scrape multiple articles and tag them with a category label.
        Returns list of {'text': ..., 'category': ...} dicts.
        """
        training_data = []
        for url in urls:
            logger.info(f"Scraping: {url}")
            article = self.scrape_article(url)
            if article and article.get('content'):
                text = f"{article['title']} {article['content']}".strip()
                training_data.append({'text': text, 'category': category})
        logger.info(f"Collected {len(training_data)} articles for category '{category}'")
        return training_data

    def save_to_csv(self, data: List[Dict[str, str]], output_path: str):
        """Save collected articles to a CSV file."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fieldnames = ['text', 'category']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Saved {len(data)} articles to {output_path}")


if __name__ == '__main__':
    # Example usage (requires network access)
    scraper = NewsScraper(delay=1.0)
    print("NewsScraper initialized. Use scrape_article(url) to scrape articles.")
    print("Example RSS feeds:")
    print("  Sports: http://rss.cnn.com/rss/edition_sport.rss")
    print("  Technology: https://feeds.feedburner.com/TechCrunch")
