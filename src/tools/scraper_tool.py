""" Web scraper Tool using httpx and BeautifulSoup """
from bs4 import BeautifulSoup
from typing import Optional
import httpx
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BlogScraper:
    """ Async web scraper using httpx for fast fetching """
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
        
    async def scrape_blog_content(self, url: str) -> Optional[str]:
        logger.info(f"Scraping blog content from: {url}")
        try:
            # We use r.jina.ai as a high-quality, bot-bypass proxy
            # It returns clean markdown which is perfect for LLMs
            jina_url = f"https://r.jina.ai/{url}"
            
            if not self.client:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as tmp_client:
                    response = await tmp_client.get(jina_url)
            else:
                response = await self.client.get(jina_url)
                
            response.raise_for_status()
            content = response.text

            if content.strip():
                logger.info(f"Successfully scraped {len(content)} characters from {url} via Jina")
                return content
            
            logger.warning(f"Jina returned empty content for {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url} via Jina: {str(e)}")
            # Fallback to direct httpx if Jina fails
            try:
                logger.info(f"Falling back to direct scraping for {url}")
                if not self.client:
                    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as tmp_client:
                        response = await tmp_client.get(url)
                else:
                    response = await self.client.get(url)
                
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                for s in soup(["script", "style", "nav", "header", "footer", "aside"]): s.decompose()
                main = soup.find('article') or soup.find("main") or soup.find('body')
                if main:
                    text = main.get_text(separator='\n', strip=True)
                    return '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            except Exception as e2:
                logger.error(f"Fallback scraping also failed for {url}: {str(e2)}")
            return None


async def scrape_blog(url: str) -> Optional[str]:
    """ Convenience function to scrape a blog post """
    async with BlogScraper() as scraper:
        return await scraper.scrape_blog_content(url)

async def scrape_blog_sync(url: str) -> Optional[str]:
    """ Synchronous wrapper for scraping """
    return await scrape_blog(url)
