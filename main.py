import os
from utils.logger import logger
from data.data import TextToCsv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    en_file = os.path.join(BASE_DIR, 'data/english.txt')
    np_file = os.path.join(BASE_DIR, 'data/nepali.txt')
    target_file = os.path.join(BASE_DIR, 'data/data.csv')
    if not os.path.exists(target_file):
        t2c = TextToCsv(en_file, np_file, target_file)
        logger.info('Data written successfully!')
    else:
        logger.info('Data already exists!')