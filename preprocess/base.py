import os
import re
import pickle
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PreprocessBase:
    def __init__(self, config)  -> None:
        self._src_lang_header = config.SRC_LANG_HEADER
        self._dest_lang_header = config.DEST_LANG_HEADER
        self._sequence_length = config.MAX_SEQ_LEN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # The data may have some 'unicode' encoding.
    # We need to change the encoding for processing of the data.
    def change_encoding(self, col):
        return col.str.encode('utf-8', errors='ignore').str.decode('utf-8')
    
    def clean_punct(self, sentence):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|$|.|!]',r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/।]',r' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence.lower()

    def read_data(self, data):
        src_lang = self.change_encoding(data[self._src_lang_header].map(str)).map(self.clean_punct)
        dest_lang = self.change_encoding(data[self._dest_lang_header].map(str)).map(self.clean_punct)
        return src_lang.tolist(), dest_lang.tolist()
    
    def load_vocab(self, w2i_file, i2w_file):
        with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
        with open(i2w_file, 'rb') as file: index_to_word = pickle.load(file)
        return word_to_index, index_to_word
    
    def padding(self, sentence):
        if len(sentence) > (self._sequence_length - 2):
            padded_sent = ['<SOS>'] + sentence[:self._sequence_length - 2] + ['<EOS>']
        else:
            padded_sent = ['<SOS>'] +  sentence + ['<EOS>'] + ['<PAD>']*(self._sequence_length - len(sentence) - 2)
        return padded_sent
    
    def all_text_to_index(self, data, w2i_file):
        with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
        idx = [
            [
                word_to_index.get(word, word_to_index['<UNK>']) 
                for word in self.padding(sentence.split())
            ]
            for sentence in data
        ]
        return idx
    
    def all_text_to_tensors(self, data, w2i_file):
        idx = self.all_text_to_index(data, w2i_file)
        tensors = torch.tensor(idx, dtype=torch.long, device=self.device)
        return tensors
    
    def single_text_to_tensors(self, sentence, w2i_file):
        with open(w2i_file, 'rb') as file: word_to_index = pickle.load(file)
        idx = [
            word_to_index.get(word, word_to_index['<UNK>']) 
            for word in self.padding(sentence.split())
        ]
        tensor = torch.tensor(idx, dtype=torch.long, device=self.device)
        return tensor