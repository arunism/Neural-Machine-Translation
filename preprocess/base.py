import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PreprocessBase:
    def __init__(self, config)  -> None:
        self._src_lang_header = config.SRC_LANG_HEADER
        self._dest_lang_header = config.DEST_LANG_HEADER
        self._output_path = os.path.join(BASE_DIR, config.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
    
    # The data may have some 'unicode' encoding.
    # We need to change the encoding for processing of the data.
    def change_encoding(self, col):
        return col.str.encode('utf-8', errors='ignore').str.decode('utf-8')
    
    def clean_punct(self, sentence):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|$|.|!]',r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence

    def read_data(self, data):
        src_lang = self.change_encoding(data[self._src_lang_header].map(str)).map(self.clean_punct)
        dest_lang = self.change_encoding(data[self._dest_lang_header].map(str)).map(self.clean_punct)
        return src_lang.tolist(), dest_lang.tolist()