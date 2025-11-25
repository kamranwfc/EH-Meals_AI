"""
EH Meals API Data Converter - OPTIMIZED VERSION
Improvements: Concurrent requests, connection pooling, batch processing
"""

import requests
import json
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

class OptimizedMealsDataConverter:
    def __init__(self, api_base_url: str, api_key: str, max_workers: int = 10):
        """
        Initialize the converter with concurrent request capability
        
        Args:
            api_base_url: Your API URL
            api_key: Your API key
            max_workers: Number of concurrent requests (default: 10)
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.max_workers = max_workers
        self.all_meals = []
        
        # Create a session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({"X-Api-Key": api_key})
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def fetch_page(self, page: int, per_page: int = 100) -> tuple[int, List[Dict], bool]:
        """
        Fetch a single page of meals
        
        Returns:
            (page_number, meals_list, is_last_page)
        """
        try:
            url = f"{self.api_base_url}/meals?per_page={per_page}&page={page}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 401:
                logger.error("Unauthorized: Invalid API key")
                return (page, [], True)
            if response.status_code == 429:
                logger.warning(f"Rate limit hit on page {page}, retrying after 1s")
                time.sleep(1)
                return (page, [], True)
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', False):
                logger.warning(f"API returned success=false for page {page}")
                return (page, [], True)
            
            meals_data = data.get('meals', {})
            page_meals = meals_data.get('data', [])
            
            current_page_num = meals_data.get('current_page', page)
            last_page = meals_data.get('last_page', page)
            is_last = current_page_num >= last_page
            
            return (page, page_meals, is_last)
            
        except Exception as e:
            logger.error(f"Error fetching page {page}: {str(e)}")
            return (page, [], True)
    
    def fetch_all_meals_concurrent(self, per_page: int = 100) -> List[Dict]:
        """
        Fetch all meals using concurrent requests
        
        Args:
            per_page: Number of meals per page
        
        Returns:
            List of all meals
        """
        start_time = time.time()
        
        # First, get the first page to know total pages
        _, first_page_meals, _ = self.fetch_page(1, per_page)
        
        if not first_page_meals:
            logger.error("Failed to fetch first page")
            return []
        
        # Get total pages from first request
        url = f"{self.api_base_url}/meals?per_page={per_page}&page=1"
        response = self.session.get(url, timeout=30)
        data = response.json()
        meals_data = data.get('meals', {})
        total_pages = meals_data.get('last_page', 1)
        
        logger.info(f"Starting fetch of {total_pages} pages")
        
        self.all_meals = first_page_meals
        
        if total_pages == 1:
            elapsed = time.time() - start_time
            logger.info(f"Fetched {len(self.all_meals)} meals in {elapsed:.2f}s")
            return self.all_meals
        
        # Fetch remaining pages concurrently
        pages_to_fetch = list(range(2, total_pages + 1))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all page requests
            future_to_page = {
                executor.submit(self.fetch_page, page, per_page): page 
                for page in pages_to_fetch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num, meals, _ = future.result()
                if meals:
                    self.all_meals.extend(meals)
        
        elapsed = time.time() - start_time
        logger.info(f"Fetched {len(self.all_meals)} meals in {elapsed:.2f}s ({len(self.all_meals) / elapsed:.1f} meals/s)")
        
        return self.all_meals
    
    def convert_meal_format_batch(self, meals: List[Dict]) -> List[Dict]:
        """
        Convert meals in batch with optimized processing
        
        Args:
            meals: List of raw meal data
            
        Returns:
            List of converted meals
        """
        start_time = time.time()
        
        converted_meals = []
        
        # Process in chunks for better memory efficiency
        chunk_size = 500
        total_chunks = (len(meals) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(meals))
            chunk = meals[start_idx:end_idx]
            
            for meal in chunk:
                try:
                    converted = self._convert_single_meal(meal)
                    converted_meals.append(converted)
                except Exception as e:
                    logger.error(f"Error converting meal {meal.get('title', 'Unknown')}: {str(e)}")
                    continue
        
        elapsed = time.time() - start_time
        logger.info(f"Converted {len(converted_meals)} meals in {elapsed:.2f}s")
        
        return converted_meals
    
    def _convert_single_meal(self, meal: Dict) -> Dict:
        """Optimized single meal conversion"""
        order_count = meal.get('total_orders', 0)
        last_ordered = meal.get('last_ordered_at')
        
        meal_type = meal.get('meal_type', {})
        
        # Optimized diet extraction
        meal_diets = [
            {
                'uuid': diet.get('uuid', ''),
                'diet_id': diet.get('diet_id', ''),
                'diet_details': {
                    'uuid': diet.get('diet_details', {}).get('uuid', ''),
                    'name': diet.get('diet_details', {}).get('name', '')
                }
            }
            for diet in meal.get('meal_diets', [])
            if diet.get('diet_details')
        ]
        
        return {
            'id': meal.get('id'),
            'uuid': meal.get('uuid', ''),
            'title': meal.get('title', ''),
            'description': meal.get('description', ''),
            'meal_type': {
                'uuid': meal_type.get('uuid', ''),
                'title': meal_type.get('title', ''),
                'name': meal_type.get('name', '')
            },
            'meal_diets': meal_diets,
            'image': meal.get('image'),
            'is_featured': meal.get('is_featured', 0),
            'cafeteria': meal.get('cafeteria', 0),
            'order_count': order_count,
            'last_ordered': last_ordered
        }
    
    def save_to_file(self, meals: List[Dict], output_path: str = 'data/meals_data.json'):
        """Save with optimized JSON writing"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        start_time = time.time()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(meals, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            elapsed = time.time() - start_time
            
            logger.info(f"Saved {len(meals)} meals to {output_path} ({file_size_mb:.2f} MB) in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
    
    def generate_statistics(self, meals: List[Dict]) -> Dict:
        """Generate statistics with optimized counting"""
        from collections import Counter
        
        meal_type_counts = Counter(
            meal.get('meal_type', {}).get('name', 'unknown')
            for meal in meals
        )
        
        diet_counts = Counter()
        for meal in meals:
            for diet in meal.get('meal_diets', []):
                diet_name = diet.get('diet_details', {}).get('name', '')
                if diet_name:
                    diet_counts[diet_name] += 1
        
        featured_count = sum(1 for m in meals if m.get('is_featured') == 1)
        cafeteria_count = sum(1 for m in meals if m.get('cafeteria') == 1)
        total_orders = sum(m.get('order_count', 0) for m in meals)
        
        return {
            'total_meals': len(meals),
            'meal_types': dict(meal_type_counts),
            'dietary_options': dict(diet_counts),
            'featured_meals': featured_count,
            'cafeteria_meals': cafeteria_count,
            'total_orders': total_orders,
            'avg_orders_per_meal': total_orders / len(meals) if meals else 0
        }
    
    def print_statistics(self, stats: Dict):
        """Print statistics"""
        pass  # No printing for API usage
    
    def __del__(self):
        """Clean up session"""
        if hasattr(self, 'session'):
            self.session.close()


def main():
    """Main function with timing"""
    
    API_BASE_URL = "https://staging-dashboard.effortlesslyhealth.com/api/v1"
    API_KEY = "4d67d97a4e659148982asf6f819fc5lo0z5"
    
    import sys
    if len(sys.argv) > 1:
        API_BASE_URL = sys.argv[1]
    if len(sys.argv) > 2:
        API_KEY = sys.argv[2]
    
    total_start = time.time()
    
    # Initialize with 10 concurrent workers (adjust based on API limits)
    converter = OptimizedMealsDataConverter(API_BASE_URL, API_KEY, max_workers=10)
    
    # Step 1: Fetch all meals concurrently
    try:
        converter.fetch_all_meals_concurrent(per_page=100)
    except Exception as e:
        logger.error(f"Fatal error fetching meals: {str(e)}")
        return
    
    if not converter.all_meals:
        logger.error("No meals were fetched")
        return
    
    # Step 2: Convert in batch
    converted_meals = converter.convert_meal_format_batch(converter.all_meals)
    
    if not converted_meals:
        logger.error("No meals were converted successfully")
        return
    
    # Step 3: Save to file
    converter.save_to_file(converted_meals)
    
    # Step 4: Generate statistics
    stats = converter.generate_statistics(converted_meals)
    
    total_elapsed = time.time() - total_start
    logger.info(f"Conversion complete in {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()