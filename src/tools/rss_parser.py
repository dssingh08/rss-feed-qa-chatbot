""" RSS Feed Parser Utilities """

import feedparser
from typing import List, Dict, Optional
from datetime import datetime
from src.utils.logger import setup_logger
from src.config import settings # Import settings

logger = setup_logger(__name__)


class RSSParser:
    """ RSS feed parser for extracting blog titles and metadata """
    @staticmethod
    def parse_feed(feed_url: str, max_entries: int = 10) -> List[Dict]:
        """
        Parse RSS feed and return recent blog titles
        
        Args:
            feed_url: RSS feed URL
            max_entries: Maximum number of entries to return
            
        Returns:
            List of blog entries with title, link, date, description
        """

        logger.info(f"Parsing RSS feed: {feed_url}")

        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_url}: {str(feed.bozo_exception)}")
            
            entries = []

            for entry in feed.entries[:max_entries]:
                blog_entry = {
                    "title": entry.get("title", "No Title"),
                    "link": entry.get("link", ""),
                    "description": (entry.get("summary") or entry.get("description") or "")[:200],
                    "published": entry.get("published", entry.get("updated", "Unknown date")),
                }
                entries.append(blog_entry)
                logger.debug(f"Parsed entry: {blog_entry['title']}")
            
            logger.info(f"Successfully parsed {len(entries)} entries from feed")
            return entries
        except Exception as e:
            logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
            return []
    
    @staticmethod
    def search_feed(feed_url: str, search_query: str) -> Optional[Dict]:
        """
        Search RSS feed for specific blog post
        
        Args:
            feed_url: RSS feed URL
            search_query: Search query to match against titles/descriptions
            
        Returns:
            Matched blog entry or None
        """

        logger.info(f"Searching feed {feed_url} for: {search_query}")

        try:
            feed = feedparser.parse(feed_url)
            search_lower = search_query.lower()

            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("summary", entry.get("description", ""))
                
                title_text = str(title) if title is not None else ""
                description_text = str(description) if description is not None else ""
                
                if search_lower in title_text.lower() or search_lower in description_text.lower():
                    result = {
                        "title": entry.get("title", "No Title"),
                        "link": entry.get("link", ""),
                        "description": (entry.get("summary") or entry.get("description") or ""),
                        "published": entry.get("published", entry.get("updated", "Unknown date")),
                    }
                    logger.info(f"Found matching blog: {result['title']}")
                    return result

            logger.info(f"No matching blog found for: {search_query}")
            return None
        except Exception as e:
            logger.error(f"Error searching feed: {str(e)}")
            return None
def get_company_feed(company_name: str) -> Optional[str]:
    """ Get RSS feed URL for a company """
    feed_url = settings.COMPANY_RSS_FEEDS.get(company_name.lower()) 

    if feed_url:
        logger.info(f"Found RSS feed for {company_name}: {feed_url}")
    else:
        logger.warning(f"No rss feed found for {company_name}")
    
    return feed_url
