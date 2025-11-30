import pandas as pd
import time
import random
import numpy as np
from datetime import datetime

import requests
from bs4 import BeautifulSoup

import selenium
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementClickInterceptedException

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.mongodb import MongoDBJobStore
from src.mongo_db.db_agoda import client
from src.mongo_db.db_agoda import Information
from src.mongo_db.db_agoda import Url_Recommendations
from src.mongo_db.db_agoda import Reviews_Rating
from src.mongo_db.db_agoda import Pages
from src.crawler.auxilliary_function.setup_module import random_delay, Create_WebDriver

def safe_select_text(soup, selector):
    try:
        element = soup.select_one(selector)
        return element.get_text(strip=True) if element else None
    except Exception as e:
        print(f"Error selecting {selector}: {e}")
        return None

def Get_Information_ (soup, url):
    #  --- Thông tin nhanh --- 
    Information_Sample = []
    
    Cancellation = safe_select_text(soup,'div[data-testid="activities-details-freeCancellation"] > span')
    Duration = safe_select_text(soup,'div[data-testid="activities-details-duration"] > span')
    Confirmation = safe_select_text(soup,'div[data-testid="activities-details-instantConfirmation"] > span')
    PickUp = safe_select_text(soup,'div[data-testid="activities-details-pickUp"] > span')
    MobileTicket = safe_select_text(soup,'div[data-testid="activities-details-mobileTicket"] > span')

    # --- Overview ---
    overview_element = soup.select_one("div[data-testid='activities-overview']")
    Overview = overview_element.get_text(separator="\n", strip=True) if overview_element else None

    # ---------- INFORMATION SAMPLE -------------
    Information_Sample.extend([Cancellation, Duration, Confirmation, PickUp, MobileTicket, Overview])
       # --- Tạo document để lưu vào MongoDB ---
       
    # --- Tìm url_id từ Pages collection ---
    page_doc = Pages.find_one({"URL": url})
    url_id = page_doc["_id"] if page_doc else None
    if not page_doc:
        print(f"Không tìm thấy url_id cho {url}")

    document = {
        "Source_URL": url,
        "Cancellation": Cancellation,
        "Duration": Duration,
        "Confirmation": Confirmation,
        "Pick_Up": PickUp,
        "Mobile_Ticket": MobileTicket,
        "Overview": Overview,
        "url_id": url_id
    }
    try:
        Information.insert_one(document)
        # print(f"Inserted Information for {url}")
    except Exception as e:
        print(f"Failed to insert Information for {url}: {e}")

def Get_URLs_Recommendation (soup, url):
    # --- Recommend URLs ---
    Recommends = soup.select("a[data-testid='activities-card-content']")
    Recommend_URLs = [recommend.get('href') for recommend in Recommends] if Recommends else []
        
    # --- Recommend Titles ---
    Titles = soup.select("div[data-testid='activities-card-wrapper-popular'] > div[style='height: 220px; max-height: 220px;'] > h3")
    Recommend_Titles = [title.get_text(strip=True) for title in Titles] if Titles else []

    # -- Recommend Badges ---
    Badges = soup.select("div[data-testid='activity-deals-badges']")
    Recommend_Badge = [badge.get_text(strip=True) for badge in Badges] if Badges else []

    # --- Recommend Final Price ---
    Prices = soup.select("div[data-testid='activities-card-total-price'] div[data-testid='activities-price-container'] > div[data-testid='activities-price']")
    Recommend_Prices = [price.get_text(strip=True) for price in Prices] if Prices else []
    
    # --- Tìm url_id từ Pages ---
    page_doc = Pages.find_one({"URL": url})
    url_id = page_doc["_id"] if page_doc else None
    
    data = {
        "Source_URL": url,
        "Recommend_URLs": Recommend_URLs if Recommend_URLs else None,
        "Recommend_Titles": Recommend_Titles if Recommend_Titles else None,
        "Recommend_Badges": Recommend_Badge if Recommend_Badge else None,
        "Recommend_Prices": Recommend_Prices if Recommend_Prices else None,
        "url_id": url_id
    }
    try:
        Url_Recommendations.insert_one(data)
    except Exception as e:
        print(f"Error inserting recommendations for {url}: {e}")
    # # ---------- RECOMMEND URLS SAMPLE -------------
    # REC_URLs_Sample.append(Recommend_URLs if Recommend_URLs else None)
    # REC_URLs_Sample.append(Recommend_Titles if Recommend_Titles else None)

def Get_Reviews (soup, url):
    # --- Rating ---
    Rating_Container = soup.select('div[data-testid="activities-review-distribution-indicator-container"] > div')
    Rating = Rating_Container[2].get_text(separator='\n', strip=True) if len(Rating_Container) > 2 else None

    # Reviews
    Reviews_ = soup.select ("div[data-testid='activities-review-content']") [:2]
    Review1 = Reviews_[0].get_text(separator='\n', strip=True) if len(Reviews_) > 0 else None
    Review2 = Reviews_[1].get_text(separator='\n', strip=True) if len(Reviews_) > 1 else None

    # ---------- REVIEWS SAMPLE -------------
    # Reviews_Sample.extend([Rating, Review1, Review2])
    
    # --- Tìm url_id từ Pages ---
    page_doc = Pages.find_one({"URL": url})
    url_id = page_doc["_id"] if page_doc else None
    
    data = {
        "Source_URL": url,
        "Rating": Rating,
        "Review_1": Review1,
        "Review_2": Review2,
        "url_id": url_id
    }
    try:
        Reviews_Rating.insert_one(data)
    except Exception as e:
        print(f"Error inserting reviews for {url}: {e}")
    
# def Preparation ():    
#     options = Options()
#     # Bỏ comment nếu muốn chạy headless (ẩn trình duyệt)
#     # options.add_argument('--headless')

#     # Các option giúp chạy ổn định trên macOS
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')

#     # Giúp ẩn dấu hiệu tự động hóa
#     options.add_argument('--disable-blink-features=AutomationControlled')

#     # Tắt popup, extension
#     options.add_argument('--disable-extensions')
#     options.add_argument('--disable-popup-blocking')

#     # Giả lập user-agent trình duyệt thật
#     options.add_argument(
#         "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
#         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
#     )

#     # Tắt GPU cho máy không dùng đồ họa hoặc headless
#     options.add_argument('--disable-gpu')

#     # Cỡ cửa sổ trình duyệt
#     options.add_argument('--window-size=1920,1080')

#     driver = webdriver.Chrome(options=options)

#     # Ẩn navigator.webdriver giúp tránh bị phát hiện bot
#     driver.execute_cdp_cmd(
#         'Page.addScriptToEvaluateOnNewDocument',
#         {
#             'source': '''
#                 Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
#             '''
#         }
#     )

#     # Ví dụ random delay mô phỏng người dùng
#     time.sleep(random.uniform(2,5))
#     return driver

def wait_for_main_elements(driver):
    WebDriverWait(driver, 12).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='activities-overview']"))
    )
    WebDriverWait(driver, 12).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='activities-card-wrapper-popular'] > div[style='height: 220px; max-height: 220px;'] > h3"))
    )
    WebDriverWait(driver, 12).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='detail-page-exploring-carousel']"))
    )

def wait_for_reviews(driver):
    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='activities-review-distribution-indicator-container']"))
        )
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='activities-review-content']"))
        )
        return True
    except:
        return False
    
def Scraping_Details (s_batches):
    driver = Create_WebDriver ()       
    failed_urls = []
    for idx, batch in enumerate (s_batches):
        success = False
        for attempt in range(3):
            try:
                driver.get (batch)
                wait_for_main_elements(driver)
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}/3 failed for {batch}: {e}")
                time.sleep(3)  # delay nhẹ giữa các lần thử lại
                
        if not success:
            print(f"Failed after 3 attempts: {batch}")
            failed_urls.append(batch)
            continue  # bỏ qua bước lấy dữ liệu cho URL lỗi
        
        has_reviews = wait_for_reviews(driver)

        # Lúc này mới lấy soup (page source đã đầy đủ)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        if has_reviews:
            Get_Reviews(soup, batch)
        else:
            print(f"No reviews on: {batch}")
            
        Get_Information_ (soup, batch)
        Get_URLs_Recommendation (soup, batch)
        
        print(f"Done {idx+1}/{len(s_batches)}: {batch}")

        if (idx + 1) % 5 == 0:
                delay = random.uniform(5, 8)
                print(f"Sleeping longer for {delay:.2f}s...")
                time.sleep(delay)
        else:
                random_delay(2, 5)
                
    driver.quit ()
    return failed_urls
    