import os
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from utils.logger import logger
from utils.split_data import train_test_split
from preprocess import PreprocessTrain, PreprocessEval
from models import LstmModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Trainer:
    def __init__(self) -> None:
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
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
        self.src_tensor = self.train_data_obj.text_to_tensor(self.train_src_data, self.src_w2i_file)
        self.dest_tensor = self.train_data_obj.text_to_tensor(self.train_dest_data, self.dest_w2i_file)

        self.src_vocab_size = len(self.src_w2i)
        self.dest_vocab_size = len(self.dest_w2i)
        self.data_size = len(self.src_tensor)
        self.get_model()

    def load_data(self) -> None:
        self.train_data_obj = PreprocessTrain(self.config)
        self.eval_data_obj = PreprocessEval(self.config)
        self.train_data, self.eval_data = train_test_split(self.data_path)
    
    def get_model(self) -> None:
        if self.config.MODEL.lower() == 'lstm':
            self.model = LstmModel(self.config, self.src_vocab_size, self.dest_vocab_size, self.dest_vocab_size)
        else:
            logger.info(f'{self.config.MODEL} is not supported!')
    
    def train(self):
        ignore_index = self.dest_w2i['<PAD>']
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # batches = len(self.train_data) // self.config.BATCH_SIZE
        epoch_loss = 0.0
        for epoch in range(self.config.EPOCHS):
            print(f'Epoch: {epoch+1}/{self.config.EPOCHS}')
            self.model.eval()
            self.model.train(True)
            # loss = self.model(self.src_tensor, self.dest_tensor, tf=self.config.TEACHER_FORCING)
            batch = 0
            # for i in range(0, self.data_size, self.batch_size):
            for i in tqdm(range(0, 100, self.batch_size)):
                batch += 1
                self.src_tensor = self.src_tensor[:100]
                self.dest_tensor = self.dest_tensor[:100]
                self.data_size = 100
                src_batch = self.src_tensor[i:i+self.batch_size] if i+self.batch_size < self.data_size else self.src_tensor[i:]
                target_batch = self.dest_tensor[i:i+self.batch_size] if i+self.batch_size < self.data_size else self.dest_tensor[i:]
                # src_batch = torch.cat(src_batch, dim=1)
                # target_batch = torch.cat(target_batch, dim=1)
                output = self.model(src_batch, target_batch, tf=self.config.TEACHER_FORCING)
                output = output.view(-1, output.size(2))
                target = target_batch.reshape(-1)
                optimizer = self.model.get_optimizer(self.config.OPTIMIZER, self.model)
                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                # restrict gradients from exploding
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                optimizer.step()
                epoch_loss += loss.item()
            print(output)
            print(output.shape)