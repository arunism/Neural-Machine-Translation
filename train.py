import config
from trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(config)
    trainer.train()