from icrawler.builtin import BingImageCrawler
import os

# Your exact drive path
SAVE_DIR = r'E:\dataset_of_capstone\dark_circles_raw'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- THE SURGICAL SEARCH LIST ---
# These terms force the search engine to look for clinical skin photos
medical_queries = [
    "periorbital hyperpigmentation dermatology patient",
    "tear trough deformity clinical photo",
    "infraorbital dark circles skin condition",
    "under eye melanin hyperpigmentation closeup",
    "severe dark circles under eyes before treatment"
]

for query in medical_queries:
    print(f"\n[Scraping Clinical Images from Bing: {query}]")
    bing_crawler = BingImageCrawler(storage={'root_dir': SAVE_DIR})
    
    # filters={'type': 'photo', 'size': 'large'} is the magic bullet to stop getting maps/tractors
    bing_crawler.crawl(
        keyword=query, 
        filters={'type': 'photo', 'size': 'large'}, 
        max_num=150
    )

print("\n✅ Scraping Complete. Check your folder.")