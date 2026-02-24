# data/scraper.py
# Downloads images for all categories using Bing image crawler
# Usage: python3 data/scraper.py
# Usage (single category): python3 data/scraper.py --category "News"

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
from icrawler.builtin import BingImageCrawler
from config import DATASET_PATH, CATEGORIES

# Search queries per category — edit these to get better images
SEARCH_TERMS = {
    "Automotive":                   ["luxury sports car showroom", "car interior dashboard", "bike scenic road"],
    "Books and Magazines":          ["stack of hardcover books", "modern magazine cover", "library reading area"],
    "E-commerce":                   ["online shopping checkout page", "ecommerce product white background", "best selling product listing"],
    "Education":                    ["students online class", "university classroom projector", "notebook study desk"],
    "Entertainment":                ["movie poster concept art", "music album promotional poster", "theatre audience"],
    "Finance":                      ["banking mobile app", "stock market trading chart", "currency bills wallet"],
    "Food and Dining":              ["gourmet meal plating", "street food vendor market", "home cooked food flat lay"],
    "Government and Public Services":["voter registration form", "government building official", "public service sign"],
    "Motivational Posters":         ["bold motivational quote design", "typography inspiring poster", "minimalist motivational text"],
    "Nature":                       ["autumn forest path", "tropical beach sunset", "mountain river valley"],
    "News":                         ["press conference podium", "breaking news tv", "newspaper headlines front page"],
    "Productivity Tools":           ["digital planner template", "task management board", "online calendar tool"],
    "Quotes (Plain Cards)":         ["simple black text white background quote", "plain motivational words minimal", "clean quote card"],
    "Receipts":                     ["restaurant bill receipt closeup", "grocery store receipt", "digital payment screenshot"],
    "Recipes":                      ["step by step recipe instructions", "recipe book open table", "food blog recipe layout"],
    "Resources":                    ["pdf file icon set", "downloadable infographic", "document library preview"],
    "Ronaldo (Celebrities)":        ["cristiano ronaldo press event", "ronaldo football action shot", "ronaldo training"],
    "Social Media Screens":         ["instagram feed layout screenshot", "twitter timeline screenshot", "social media profile app"],
    "Sports":                       ["soccer stadium aerial view", "basketball game action shot", "athletes celebrating victory"],
    "Technology":                   ["ai robot concept illustration", "multiple computer monitors workstation", "tech startup office"],
}

def scrape_category(category: str, max_per_query: int = 50):
    save_dir = os.path.join(DATASET_PATH, category)
    os.makedirs(save_dir, exist_ok=True)

    queries = SEARCH_TERMS.get(category)
    if not queries:
        print(f"⚠️  No search terms defined for: {category}")
        return

    print(f"\n📥 Scraping: {category}")
    for query in queries:
        print(f"   🔍 Query: '{query}'")
        crawler = BingImageCrawler(storage={'root_dir': save_dir})
        try:
            crawler.crawl(keyword=query, max_num=max_per_query)
        except Exception as e:
            print(f"   ⚠️  Failed: {e}")

    total = len([f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    print(f"   ✅ {category} now has {total} images")


def scrape_all(max_per_query: int = 50):
    print(f"🚀 Scraping all {len(CATEGORIES)} categories...")
    for cat in CATEGORIES:
        scrape_category(cat, max_per_query)
    print("\n🎉 All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default=None,
                        help='Single category to scrape. Leave empty to scrape all.')
    parser.add_argument('--max', type=int, default=50,
                        help='Max images per search query (default: 50)')
    args = parser.parse_args()

    if args.category:
        scrape_category(args.category, args.max)
    else:
        scrape_all(args.max)
