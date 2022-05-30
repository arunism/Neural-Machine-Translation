import os
import pickle
from collections import Counter
from utils.logger import logger
from preprocess.base import PreprocessBase

class PreprocessTrain(PreprocessBase):
    def __init__(self, config) -> None:
        super(PreprocessTrain, self).__init__(config)
        self._min_word_count = config.MIN_WORD_COUNT if config.MIN_WORD_COUNT else 1
    
    def build_vocab(self, data, w2i_file, i2w_file):
        if os.path.exists(w2i_file) and os.path.exists(i2w_file):
            logger.info('Loading vocab from existing files...')
            w2i, i2w = self.load_vocab(w2i_file, i2w_file)
            return w2i, i2w
        
        logger.info('Building vocab...')
        words = [word for sentence in data for word in sentence.split()]
        word_count = Counter(words)
        sorted_words = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
        words = [item[0] for item in sorted_words if item[1] >= self._min_word_count]
        vocab = ['<SOS>', '<EOS>', '<PAD>', '<UNK>'] + words

        w2i, i2w = dict(), dict()
        for i, w in enumerate(vocab):
            w2i[w] = i
            i2w[i] = w
        with open(w2i_file, 'wb') as file: pickle.dump(w2i, file)
        with open(i2w_file, 'wb') as file: pickle.dump(i2w, file)
        return w2i, i2w