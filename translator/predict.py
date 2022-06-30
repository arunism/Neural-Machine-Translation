import os
import torch
from glob import glob
from trainer import Trainer
from preprocess import PreprocessEval

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Translator:
    def __init__(self, config) -> None:
        self.config = config
        self.max_seq_len = self.config.MAX_SEQ_LEN
        self._output_path = os.path.join(BASE_DIR, self.config.OUTPUT_PATH)
        self._data_path = os.path.join(self._output_path, 'data')
        self._model_path = os.path.join(self._output_path, 'models')
        self.src_w2i_file = os.path.join(self._data_path, config.SRC_W2I_FILE)
        self.dest_w2i_file = os.path.join(self._data_path, config.DEST_W2I_FILE)
        self.src_i2w_file = os.path.join(self._data_path, config.SRC_I2W_FILE)
        self.dest_i2w_file = os.path.join(self._data_path, config.DEST_I2W_FILE)
        self.preprocess_obj = PreprocessEval(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        self.model_exists()
    
    def model_exists(self):
        if not os.path.exists(self._model_path) or not os.listdir(self._model_path):
            trainer = Trainer(self.config)
            trainer.train()