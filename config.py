# You can choose between ['lstm', 'lstm_atten', 'rcnn', 'textcnn', 'transformer']
MODEL = 'lstm'
EPOCHS = 2
LEARNING_RATE = 0.001
# You can choose between ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']
OPTIMIZER = 'adam'
SEQUENCE_LEN = 100
EMBED_SIZE = 200
HIDDEN_SIZE = [256]
BATCH_SIZE = 128
VOCAB_SIZE = None
DATA_PATH = 'data/data.csv'
OUTPUT_PATH = 'results/'
TRAIN_TEST_SPLIT_RATIO = 0.8