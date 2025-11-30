import time

from src.crawler.auxilliary_function.metadata import get_metadata

from src.mongo_db.db_agoda import Pages
from src.mongo_db.db_agoda import Agoda_Activities_Reviews

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import ElementClickInterceptedException

def Scraping_Reviews (driver, url, object_id):
    # authors = driver.find_elements (By.XPATH, '//div[@class="css-1dbjc4n r-13awgt0 r-9aw3ui"]') : Không có
    # Số lần tối đa thử click lại 1 nút
    MAX_RETRIES_PER_BUTTON = 5

    try:
        expand_buttons = driver.find_elements(
            By.XPATH,
            '//div[@class="aaa1e-box aaa1e-fill-inherit aaa1e-text-inherit aaa1e-mt-8      "]'
            '//p[@class="sc-gwsNht Typographystyled__TypographyStyled-sc-1uoovui-0 isoMbC hiXmOF"]//button'
        )

        for btn in expand_buttons:
            while True:
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                    time.sleep(1)  # cho scroll xong
                    btn.click()
                    time.sleep(5)
                    break
                except (ElementClickInterceptedException, StaleElementReferenceException):
                    time.sleep(2)

    except NoSuchElementException:
        # print("Không tìm thấy nút mở rộng nào.")
        pass
        
    titles = driver.find_elements (By.XPATH, '//div[@data-testid="activities-review-content"]//p[@class="sc-gwsNht Typographystyled__TypographyStyled-sc-1uoovui-0 bfilTo UwrgA"]')

    contents = driver.find_elements (By.XPATH, '//div[@class="aaa1e-box aaa1e-fill-inherit aaa1e-text-inherit aaa1e-mt-8      "]')
        
    per_ratings = driver.find_elements (By.XPATH, '//div[@data-testid="review-card-title"]//span[@class="sc-gwsNht Typographystyled__TypographyStyled-sc-1uoovui-0 hQhxAT bmjHjQ"]')
        
    time_reviews = driver.find_elements (By.XPATH, '//div[@class="aaa1e-box aaa1e-fill-inherit aaa1e-text-inherit aaa1e-flex aaa1e-flex-row      "]//div[@class="aaa1e-box aaa1e-fill-inherit aaa1e-text-inherit aaa1e-flex aaa1e-flex-col aaa1e-justify-evenly      "]//span[@class="sc-gwsNht Typographystyled__TypographyStyled-sc-1uoovui-0 isoMbC oWowW"]')

    Sources = driver.find_elements (By.XPATH, '//div[@class="aaa1e-box aaa1e-fill-inherit aaa1e-text-inherit aaa1e-mt-16      "]//span[@class="sc-gwsNht Typographystyled__TypographyStyled-sc-1uoovui-0 isoMbC ktgvnM"]')
        
    # Translated_Titles = driver.find_elements (By.XPATH, '//div[@data-element-name="activities-review-translated"]//p[@class="sc-hoLldG sc-jZhnRx ceImgB iBnGae"]')
    
    # Translated_Contents = driver.find_elements (By.XPATH, '//div[@data-element-name="activities-review-translated"]//p[@class="sc-hoLldG sc-jZhnRx jQsqdf ehUZtX"]')
    
    num_reviews = min(
        len(titles),
        len(contents),
        len(per_ratings),
        len(time_reviews),
        len (Sources)
    )
    
    for i in range(num_reviews):
        review_data = {
            'Activity_Id': object_id,
            "Title": titles[i].text.strip(),
            "Content": contents[i].text.strip(),
            "Per_Rating": per_ratings[i].text.strip(),
            "Time_Review": time_reviews[i].text.strip(),
            "Source": Sources[i].text.strip(),
        }
        metadata = get_metadata (url)
        if metadata:
            review_data.update (metadata)
            
        try:
            Agoda_Activities_Reviews.insert_one(review_data)
            # print (review_data)
        except Exception as e:
            print(f"Error inserting reviews for {url}: {e}")
            
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
    
def scroll_to_bottom(driver, step_delay=0.3, max_scrolls=30):
    """
    Cuộn xuống dưới cùng của trang web một cách tuần tự.
    Thường dùng để buộc các phần tử lazy-load xuất hiện (ví dụ: nút Next, comment, review, ...)
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(step_delay)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            # print(f"Đã cuộn tới đáy trang sau {i+1} lần.")
            break
        last_height = new_height
    else:
        print("Đã cuộn tối đa mà chưa đến đáy trang.")

            