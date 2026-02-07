import logging
import os
import sys
from datetime import datetime
import copy

# Định nghĩa các mã màu ANSI
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Màu nền hoặc màu đậm cho các level cụ thể
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"

class ColoredFormatter(logging.Formatter):
    """
    Formatter tùy chỉnh để thêm màu sắc dựa trên Log Level.
    """
    
    date_fmt = '%Y-%m-%d %H:%M:%S'
    
    # Định dạng cho từng level
    FORMATS = {
        logging.DEBUG:    f"{Colors.grey}%(asctime)s{Colors.RESET} | {Colors.grey}%(levelname)-8s{Colors.RESET} | %(message)s",
        logging.INFO:     f"{Colors.CYAN}%(asctime)s{Colors.RESET} | {Colors.GREEN}%(levelname)-8s{Colors.RESET} | %(message)s",
        logging.WARNING:  f"{Colors.CYAN}%(asctime)s{Colors.RESET} | {Colors.YELLOW}%(levelname)-8s{Colors.RESET} | {Colors.YELLOW}%(message)s{Colors.RESET}",
        logging.ERROR:    f"{Colors.CYAN}%(asctime)s{Colors.RESET} | {Colors.RED}%(levelname)-8s{Colors.RESET} | {Colors.RED}%(message)s{Colors.RESET}",
        logging.CRITICAL: f"{Colors.CYAN}%(asctime)s{Colors.RESET} | {Colors.BOLD}{Colors.RED}%(levelname)-8s{Colors.RESET} | {Colors.BOLD}{Colors.RED}%(message)s{Colors.RESET}"
    }

    def format(self, record):
        # Sao chép record để không ảnh hưởng đến các handler khác (như FileHandler)
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt)
        return formatter.format(record)

class Logger:
    """
    - Ghi ra file: Văn bản thuần (Dễ đọc bằng editor).
    - Ghi ra màn hình: Có màu sắc (Dễ quan sát).
    """
    def __init__(self, output_dir, name="DeepLearning_Project"):
        self.output_dir = output_dir
        self.name = name
        self.logger = self._setup_logger()

    def _setup_logger(self):
        # Tạo thư mục nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tạo tên file log theo timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.output_dir, f"log_{self.name}_{timestamp}.txt")

        # Khởi tạo logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Xóa các handler cũ nếu có (tránh duplicate log khi gọi lại class)
        if logger.hasHandlers():
            logger.handlers.clear()

        # --- 1. FILE HANDLER (PLAIN TEXT) ---
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # --- 2. STREAM HANDLER (COLORED) ---
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter())
        logger.addHandler(stream_handler)

        logger.info(f" Log file created at: {log_filename}")
        return logger

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)