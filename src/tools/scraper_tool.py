""" Web scraper Tool using Playwright """
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from typing import Optional
import asyncio
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BlogScraper:
    """ Async web scraper using Playwright for dynamic content """
    def __init__(self):
        self.playwright = None
        self.browser = None

    
    async def __aenter__(self):
        """ Async context manager entry """
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ Async context manager exit """
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
    async def scrape_blog_content(self, url: str) -> Optional[str]:
        """
        Scraper blog content from URL

        Args:
            url: Blog post URL
            
        Returns:
            Cleaned blog content or None
        """
        logger.info(f"Scraping blog content from: {url}")
        try:
            if not self.browser:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(headless=True)
            page = await self.browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=30000)

            content = await page.content()
            await page.close()

            soup = BeautifulSoup(content, 'html.parser')

            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            main_content = (
                soup.find('article') or
                soup.find("main") or
                soup.find('div', class_=['content', 'post', 'article']) or
                soup.find('body')
            )

            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

                logger.info(f"Successfully scraped {len(text)} characters from {url}")
                return text
            else:
                logger.warning(f"Could not find main content in {url}")
                return None
        except PlaywrightTimeout:
            logger.error(f"Timeout while scraping {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None



async def scrape_blog(url: str) -> Optional[str]:
    """ Convenience function to scrape a blog post """
    async with BlogScraper() as scraper:
        return await scraper.scrape_blog_content(url)


def scrape_blog_sync(url: str) -> Optional[str]:
    """ Synchronous wrapper for scraping """
    try: 
        return asyncio.run(scrape_blog(url))
    except Exception as e:
        logger.error(f"Error in sync scrape: {str(e)}")
        return None
