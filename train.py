import os
import config
from utils.split_data import train_test_split
from preprocess import PreprocessTrain, PreprocessEval

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Train:
    def __init__(self) -> None:
        self.config = config
        self.data_path = self.config.DATA_PATH
        self.train_data_obj = None
        self.eval_data_obj = None
        self.model = None
        self.train_data = None
        self.eval_data = None

        self.load_data()
        self.train_src_lang, self.train_dest_lang = self.train_data_obj.read_data(self.train_data)
        print(self.train_src_lang[:5])
        print(self.train_dest_lang[:5])

    def load_data(self) -> None:
        self.train_data_obj = PreprocessTrain(self.config)
        self.eval_data_obj = PreprocessEval(self.config)
        self.train_data, self.eval_data = train_test_split(self.data_path)
        print(self.train_data.head())


if __name__ == '__main__':
    trainer = Train()