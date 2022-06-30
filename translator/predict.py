import os
import pickle
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
        self.get_model()
    
    def model_exists(self):
        trainer = Trainer(self.config)
        self.model = trainer.model
        if not os.path.exists(self._model_path) or not os.listdir(self._model_path):
            trainer.train()
    
    def get_model(self):
        modles_list = glob(os.path.join(self._model_path, '*'))
        most_recent_model = max(modles_list, key=os.path.getctime)
        self.model.load_state_dict(torch.load(most_recent_model, map_location=self.device))
        self.model.eval()
    
    def predict(self, sentence):
        with open(self.dest_i2w_file, 'rb') as file: dest_i2w = pickle.load(file)
        sent_tensor = self.preprocess_obj.single_text_to_tensor(sentence, self.src_w2i_file)
        with torch.no_grad(): hidden, cell = self.model.encoder(sent_tensor)
        outputs = [0]
        for i in range(self.max_seq_len):
            prev_word = torch.LongTensor([outputs[-1]]).to(self.device)
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(prev_word, hidden, cell)
                prediction = output.argmax(1).item()
            outputs.append(prediction)
            if dest_i2w[prediction] == '<EOS>': break
        
        token_prediction = [dest_i2w[idx] for idx in outputs]
        translation = ' '.join(token_prediction)
        return translation