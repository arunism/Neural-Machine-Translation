import os
import config
from utils.split_data import train_test_split
from preprocess import PreprocessTrain, PreprocessEval
from models import LstmModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(self) -> None:
        self.config = config
        self.data_path = self.config.DATA_PATH
        self._output_path = os.path.join(BASE_DIR, config.OUTPUT_PATH)
        if not os.path.exists(self._output_path): os.makedirs(self._output_path)
        self.src_w2i_file = os.path.join(self._output_path, config.SRC_W2I_FILE)
        self.dest_w2i_file = os.path.join(self._output_path, config.DEST_W2I_FILE)
        self.src_i2w_file = os.path.join(self._output_path, config.SRC_I2W_FILE)
        self.dest_i2w_file = os.path.join(self._output_path, config.DEST_I2W_FILE)
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.train_data = None
        self.eval_data = None

        self.load_data()
        self.train_src_data, self.train_dest_data = self.train_data_obj.read_data(self.train_data)
        self.src_w2i, self.src_i2w = self.train_data_obj.build_vocab(self.train_src_data, self.src_w2i_file, self.src_i2w_file)
        self.dest_w2i, self.dest_i2w = self.train_data_obj.build_vocab(self.train_dest_data, self.dest_w2i_file, self.dest_i2w_file)
        self.src_text2idx = self.train_data_obj.text_to_tensor(self.train_src_data, self.src_w2i_file)
        self.dest_text2idx = self.train_data_obj.text_to_tensor(self.train_dest_data, self.dest_w2i_file)
        # print(self.dest_text2idx[:3])
        # print(self.src_text2idx[:3])
        self.model = LstmModel(self.config, 50000)

    def load_data(self) -> None:
        self.train_data_obj = PreprocessTrain(self.config)
        self.eval_data_obj = PreprocessEval(self.config)
        self.train_data, self.eval_data = train_test_split(self.data_path)


if __name__ == '__main__':
    trainer = Train()