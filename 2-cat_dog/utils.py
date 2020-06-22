import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def is_jsonable(obj):
    """
    判断对象obj是否可json序列化
    :param obj:
    :return:
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):  # OverflowError是对象太大导致的无法序列化
        return False


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    # log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def fig_confusion_matrix(confusion, categories_list):
    # 绘制混淆矩阵。对每类的分类情况做了归一化，防止有的类数量太多颜色太深，其他类数量少颜色太浅
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    normalize_confusion = confusion / confusion.sum(1).reshape((-1,1))
    cax = ax.matshow(normalize_confusion)
    fig.colorbar(cax)

    # Set up axes
    list_categories_and_num = [f"{num}  {c}" for num,c in zip(confusion.sum(1), categories_list)]
    ax.set_xticklabels([''] + categories_list, rotation=90)
    ax.set_yticklabels([''] + list_categories_and_num)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.tight_layout()  # 适应窗口大小，否则可能有东西在窗口里画不下
    return fig


def fig_images(img):
    fig, ax = plt.subplots()
    npimg = img.numpy()
    plt_img = ax.imshow(np.transpose(npimg, (1, 2, 0)))
    fig.tight_layout()
    return fig