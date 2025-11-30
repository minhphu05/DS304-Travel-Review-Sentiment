from datetime import datetime

def get_metadata(source):
    now = datetime.now()
    metadata = {
        "SourcePlatform": source,
        "Date": now.strftime("%Y-%m-%d"),       # ví dụ: '2025-06-11'
        "Time": now.strftime("%H:%M:%S")        # ví dụ: '17:36:27'
    }
    return metadata