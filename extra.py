import json
from src.tools.rss_parser import RSSParser, get_company_feed
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def fetch_and_save_all_feeds(output_path="rss_feeds_output.json"):
    company_data = {}
    for company in ["google", "openai", "amazon", "microsoft", "meta", "anthropic"]:
        feed_url = get_company_feed(company)
        if not feed_url:
            logger.warning(f"No RSS feed found for company {company}")
            continue
        entries = RSSParser.parse_feed(feed_url, max_entries=50)
        company_data[company] = entries
        logger.info(f"Fetched {len(entries)} entries for {company}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(company_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved RSS feed data of all companies to {output_path}")

if __name__ == "__main__":
    fetch_and_save_all_feeds()
