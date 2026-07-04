import os
import sys
import datetime

class TeeLogger:
    def __init__(self, log_dir='', log_file=None, tag="default", auto_timestamp=True, rank=0):
        """
        初始化日志记录器，支持分布式环境
        
        Args:
            log_file: 日志文件路径，如果为None则自动生成
            tag: 日志标签，用于自动生成文件名
            auto_timestamp: 是否在自动生成的文件名中包含时间戳
            rank: 当前进程的rank（默认为0）
        """
        # 保存原始的stdout
        self.original_stdout = sys.stdout
        self.is_setup = False
        self.rank = rank
        self.log_dir=f'logs/{log_dir}/'
        # 确定日志文件路径
        if log_file is None:
            # 自动生成文件名
            base_name = f"log_{tag}"
            if auto_timestamp:
                timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                base_name = f"{base_name}_{timestamp}"
            log_file = f"{base_name}.log"

        self.log_file = self.log_dir+log_file

    def setup(self):
        """
        设置日志记录器，开始记录日志
        """
        if not self.is_setup:
            # 创建日志目录（如果不存在）
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # 打开日志文件（行缓冲模式）
            self.file = open(self.log_file, 'w', encoding='utf-8', buffering=1)
            
            # 重定向stdout
            sys.stdout = self
            self.is_setup = True
            
            # 输出日志文件信息
            self.original_stdout.write(f"Logging to file: {self.log_file}\n")
            self.file.write(f"Logging to file: {self.log_file}\n")
            
        return self
    
    def write(self, message):
        """
        写入消息到控制台和日志文件
        """
        if not self.is_setup:
            self.setup()
        
        # 如果是分布式环境且rank不为0，在消息前添加rank标识
        if self.rank != 0:
            # 只在新行开始时添加rank标识
            if message.startswith('\n'):
                message = message.replace('\n', f'\n[Rank {self.rank}] ', 1)
            elif message:
                message = f'[Rank {self.rank}] {message}'
        
        self.original_stdout.write(message)
        self.file.write(message)
        self.flush()
    
    def flush(self):
        """
        刷新缓冲区
        """
        if self.is_setup:
            self.original_stdout.flush()
            self.file.flush()
    
    def close(self):
        """
        关闭日志记录器，恢复原始stdout
        """
        if self.is_setup:
            self.file.close()
            sys.stdout = self.original_stdout
            self.is_setup = False

# 全局日志记录器实例
_logger_instance = None

def setup_logger(log_file=None, out_dir= "", tag="default", auto_timestamp=True, rank=0, world_size=1, log_all_ranks=True):
    """
    设置全局日志记录器，支持分布式环境
    
    Args:
        log_file: 日志文件路径，如果为None则自动生成
        tag: 日志标签，用于自动生成文件名
        auto_timestamp: 是否在自动生成的文件名中包含时间戳
        rank: 当前进程的rank（默认为0）
        world_size: 总进程数（默认为1）
        log_all_ranks: 是否所有rank都记录日志，设置为False时只有rank 0会记录
    
    Returns:
        TeeLogger实例
    """
    # 如果不是rank 0且不需要所有rank记录日志，则返回None
    if rank != 0 and not log_all_ranks:
        return None
    
    global _logger_instance
    if _logger_instance is None:
        # 如果是分布式环境且需要区分不同rank的日志文件
        if world_size > 1 and log_all_ranks:
            # 在日志文件名中添加rank信息
            if log_file is None:
                base_name = f"log_{tag}_rank{rank}"
                if auto_timestamp:
                    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
                    base_name = f"{base_name}_{timestamp}"
                log_file = f"{base_name}.log"                
            else:
                # 如果提供了log_file，在文件名中插入rank信息
                name, ext = os.path.splitext(log_file)
                log_file = f"{name}_rank{rank}{ext}"

        _logger_instance = TeeLogger(out_dir, log_file, tag, auto_timestamp, rank)

    # 记录rank和world_size信息
    logger = _logger_instance.setup()
    if logger and rank == 0:
        print(f"Starting distributed training with world_size={world_size}, rank={rank}")
    
    return logger

def get_logger():
    """
    获取全局日志记录器实例
    
    Returns:
        TeeLogger实例或None（如果未初始化）
    """
    global _logger_instance
    return _logger_instance

def close_logger():
    """
    关闭全局日志记录器
    """
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.close()
        _logger_instance = None

# 上下文管理器支持
class LoggerContext:
    def __init__(self, log_file=None, tag="default", auto_timestamp=True, rank=0, world_size=1, log_all_ranks=True):
        self.log_file = log_file
        self.tag = tag
        self.auto_timestamp = auto_timestamp
        self.rank = rank
        self.world_size = world_size
        self.log_all_ranks = log_all_ranks
    
    def __enter__(self):
        return setup_logger(self.log_file, self.tag, self.auto_timestamp, self.rank, self.world_size, self.log_all_ranks)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        close_logger()

# 便捷函数：创建上下文管理器
def logger_context(log_file=None, tag="default", auto_timestamp=True, rank=0, world_size=1, log_all_ranks=True):
    """
    创建日志记录器上下文管理器，支持分布式环境
    
    Args:
        log_file: 日志文件路径，如果为None则自动生成
        tag: 日志标签，用于自动生成文件名
        auto_timestamp: 是否在自动生成的文件名中包含时间戳
        rank: 当前进程的rank（默认为0）
        world_size: 总进程数（默认为1）
        log_all_ranks: 是否所有rank都记录日志，设置为False时只有rank 0会记录
    
    Returns:
        LoggerContext实例
    """
    return LoggerContext(log_file, tag, auto_timestamp, rank, world_size, log_all_ranks)