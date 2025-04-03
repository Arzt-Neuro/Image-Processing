import logging
import os
from datetime import datetime
import sys
import io
import shutil

class StreamLogRecorder:
    def __init__(self, log_dir='logs', verbose=False):
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 生成带有日期和时间的日志文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'log_{current_time}.log')

        # 配置日志
        self.logger = logging.getLogger('training_logger')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # 文件日志处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # 控制台日志处理器（用于覆盖打印）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)
        self.terminal_width = shutil.get_terminal_size().columns // 4
        self.verbose = verbose

        # 保存原始 stdout
        self._stdout = sys.stdout

        # 创建一个可以覆盖打印的流
        self.overwrite_stream = io.StringIO()

    def log(self, message):
        if self.verbose:
            print('\r' + ' ' * self.terminal_width, end='', file=self._stdout, flush=True)
            print('\r' + message, end='', file=self._stdout, flush=True)

        # 记录到日志文件
        self.logger.info(message)

    def final_log(self, message):
        """最终日志记录（不覆盖）"""
        # 确保最后一行完整显示
        if self.verbose:
            print("\n", end='', file=self._stdout)
            print(message, file=self._stdout)

        # 记录到日志文件和控制台
        self.logger.info(message)

# 使用示例
def training_process():
    # 创建日志记录器
    logger_printer = StreamLogRecorder('output')

    # 模拟训练过程
    max_epochs = 10
    for epoch in range(max_epochs):
        # 覆盖打印
        logger_printer.log(f"Epoch {epoch+1}/{max_epochs} - Processing...")

        # 模拟一些训练步骤
        for step in range(5):
            logger_printer.log(f"Epoch {epoch+1}/{max_epochs} - Step {step+1}/5")

        # 假设的指标
        accuracy = 0.85 + epoch * 0.01
        logger_printer.log(f"Epoch {epoch+1} - Accuracy: {accuracy:.4f}")

    # 最终日志
    logger_printer.final_log("Training completed successfully!")

# 主执行入口
if __name__ == "__main__":
    training_process()