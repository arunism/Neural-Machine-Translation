import config
from trainer import Trainer
from translator import Translator

src_text = 'के छ खबर?'

if __name__ == '__main__':
    # trainer = Trainer(config)
    # trainer.train()
    translator = Translator(config)
    prediction = translator.predict(src_text)
    print(prediction)