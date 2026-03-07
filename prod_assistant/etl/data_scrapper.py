import csv
import time
import re
import os
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class FlipkartScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_driver(self):
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = uc.Chrome(options=options, version_main=145, use_subprocess=True)
        return driver

    def _close_popup(self, driver):
        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
            time.sleep(1)
        except Exception as e:
            print(f"Error occurred while closing popup: {e}")

    def get_top_reviews(self, driver, count=2):
        """Get the top reviews from current product page."""
        try:
            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            review_blocks = soup.select("div.v1zwn21k.v1zwn26._1psv1zeb9._1psv1ze0")

            seen = set()
            reviews = []

            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break

            return " || ".join(reviews) if reviews else "No reviews found"

        except Exception as e:
            print(f"Error occurred while getting reviews: {e}")
            return "No reviews found"

    def scrape_flipkart_products(self, query, max_products=1, review_count=2):
        """Scrape Flipkart products based on a search query."""
        driver = self._create_driver()
        products = []

        try:
            search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
            driver.get(search_url)
            time.sleep(4)

            self._close_popup(driver)
            time.sleep(2)

            items = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-id]"))
            )[:max_products]

            for item in items:
                try:
                    # Get main details from search result card itself
                    try:
                        title = item.find_element(By.CSS_SELECTOR, "div.RG5Slk").text.strip()
                    except Exception:
                        title = "N/A"

                    try:
                        price = item.find_element(By.CSS_SELECTOR, "div.hZ3P6w.DeU9vF").text.strip()
                    except Exception:
                        price = "N/A"

                    try:
                        rating = item.find_element(By.CSS_SELECTOR, "div.MKiFS6").text.strip()
                    except Exception:
                        rating = "N/A"

                    try:
                        reviews_text = item.find_element(By.CSS_SELECTOR, "span.PvbNMB").text.strip()
                        match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                        total_reviews = match.group(0) if match else "N/A"
                    except Exception:
                        total_reviews = "N/A"

                    try:
                        link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                        href = link_el.get_attribute("href")
                        product_link = href if href.startswith("http") else "https://www.flipkart.com" + href
                    except Exception:
                        product_link = ""

                    match = re.findall(r"/p/(itm[0-9A-Za-z]+)", product_link)
                    product_id = match[0] if match else "N/A"

                except Exception as e:
                    print(f"Error occurred while processing item: {e}")
                    continue

                # Only go inside product page for reviews
                if "flipkart.com" in product_link:
                    try:
                        driver.get(product_link)
                        time.sleep(4)
                        self._close_popup(driver)
                        top_reviews = self.get_top_reviews(driver, count=review_count)
                    except Exception as e:
                        print(f"Error occurred while fetching reviews for product: {e}")
                        top_reviews = "No reviews found"

                    # Go back to search page so next item still works from the results list
                    try:
                        driver.back()
                        time.sleep(3)
                        self._close_popup(driver)

                        # Reload items because old Selenium references may go stale after back()
                        items = WebDriverWait(driver, 10).until(
                            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div[data-id]"))
                        )[:max_products]
                    except Exception as e:
                        print(f"Error occurred while returning to search page: {e}")
                else:
                    top_reviews = "Invalid product URL"

                products.append([product_id, title, rating, total_reviews, price, top_reviews])

        except Exception as e:
            print(f"Error occurred during scraping: {e}")

        driver.quit()
        return products

    def save_to_csv(self, data, filename="product_reviews.csv"):
        """Save the scraped product reviews to a CSV file."""
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)



