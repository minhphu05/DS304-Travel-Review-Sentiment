import time, random
from selenium.webdriver.chrome.options import Options
from selenium import webdriver

def random_delay(min_sec=2, max_sec=5):
    '''Gọi random_delay sau mỗi trang scrape'''
    time.sleep(random.uniform(min_sec, max_sec))
    
def Create_WebDriver ():  
    """Khởi tạo WebDriver với cấu hình tùy chỉnh"""  
    options = Options()
    # Bỏ comment nếu muốn chạy headless (ẩn trình duyệt)
    # options.add_argument('--headless')

    # Các option giúp chạy ổn định trên macOS
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Giúp ẩn dấu hiệu tự động hóa
    options.add_argument('--disable-blink-features=AutomationControlled')

    # Tắt popup, extension
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-popup-blocking')

    # Giả lập user-agent trình duyệt thật
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )

    # Tắt GPU cho máy không dùng đồ họa hoặc headless
    options.add_argument('--disable-gpu')

    # Cỡ cửa sổ trình duyệt
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)

    # Ẩn navigator.webdriver giúp tránh bị phát hiện bot
    driver.execute_cdp_cmd(
        'Page.addScriptToEvaluateOnNewDocument',
        {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            '''
        }
    )

    # Ví dụ random delay mô phỏng người dùng
    time.sleep(random.uniform(2,5))
    return driver