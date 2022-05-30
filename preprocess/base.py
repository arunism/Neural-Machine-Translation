import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PreprocessBase:
    def __init__(self, config)  -> None:
        self._src_lang_header = config.SRC_LANG_HEADER
        self._dest_lang_header = config.DEST_LANG_HEADER
        self._output_path = os.path.join(BASE_DIR, config.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)

    def read_data(self, data):
        src_lang = data[self._src_lang_header].map(str)
        dest_lang = data[self._dest_lang_header].map(str)
        return src_lang.tolist(), dest_lang.tolist()