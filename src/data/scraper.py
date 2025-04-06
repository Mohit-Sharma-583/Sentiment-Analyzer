import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def get_amazon_reviews(self, product_url: str, num_pages: int = 5) -> List[Dict[str, Any]]:
        """
        Scrape reviews from Amazon product pages
        
        Args:
            product_url: URL of the Amazon product
            num_pages: Number of review pages to scrape
            
        Returns:
            List of dictionaries containing review data
        """
        reviews = []
        
        try:
            for page in range(1, num_pages + 1):
                logger.info(f"Scraping page {page} of {num_pages}")
                
                # Construct review page URL
                review_url = f"{product_url}/ref=cm_cr_othr_d_paging_btm_next_{page}?ie=UTF8&reviewerType=all_reviews&pageNumber={page}"
                
                response = requests.get(review_url, headers=self.headers)
                if response.status_code != 200:
                    logger.error(f"Failed to fetch page {page}")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                review_elements = soup.find_all('div', {'data-hook': 'review'})
                
                for review in review_elements:
                    try:
                        review_data = {
                            'text': review.find('span', {'data-hook': 'review-body'}).text.strip(),
                            'rating': int(review.find('i', {'data-hook': 'review-star-rating'}).text.split('.')[0]),
                            'title': review.find('a', {'data-hook': 'review-title'}).text.strip(),
                            'date': review.find('span', {'data-hook': 'review-date'}).text.strip(),
                            'source': 'amazon',
                            'scraped_at': datetime.now().isoformat()
                        }
                        reviews.append(review_data)
                    except Exception as e:
                        logger.error(f"Error parsing review: {str(e)}")
                        continue
                
                # Random delay to avoid being blocked
                time.sleep(random.uniform(2, 5))
                
        except Exception as e:
            logger.error(f"Error scraping reviews: {str(e)}")
            
        return reviews
    
    def save_reviews(self, reviews: List[Dict[str, Any]], filename: str):
        """
        Save scraped reviews to a CSV file
        
        Args:
            reviews: List of review dictionaries
            filename: Output filename
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df = pd.DataFrame(reviews)
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(reviews)} reviews to {filename}")
        except Exception as e:
            logger.error(f"Error saving reviews: {str(e)}")

if __name__ == "__main__":
    # Example usage
    scraper = ReviewScraper()
    
    # List of product URLs to try scraping
    product_urls = [
        "https://www.amazon.com/dp/B07ZPKBL9V",  # Echo Dot
        "https://www.amazon.com/dp/B09B8V1LZ3",  # Kindle Paperwhite
        "https://www.amazon.com/dp/B08R59YH7W",  # Fire TV Stick
        "https://www.amazon.com/dp/B07FZ8S74R",  # Echo Show
        "https://www.amazon.com/dp/B07PVCVBN7"   # Fire HD Tablet
    ]
    
    all_reviews = []
    
    # Try each product URL until we get some reviews
    for product_url in product_urls:
        logger.info(f"Attempting to scrape reviews from {product_url}")
        reviews = scraper.get_amazon_reviews(product_url, num_pages=2)
        
        if reviews:
            all_reviews.extend(reviews)
            logger.info(f"Successfully scraped {len(reviews)} reviews")
            
            # If we have enough reviews, break out of the loop
            if len(all_reviews) >= 20:
                break
                
        # Add a delay between product requests
        time.sleep(random.uniform(3, 7))
    
    if all_reviews:
        # Save reviews
        scraper.save_reviews(all_reviews, "data/amazon_reviews.csv")
        logger.info(f"Total reviews scraped and saved: {len(all_reviews)}")
    else:
        logger.error("Failed to scrape any reviews from all product URLs.") 