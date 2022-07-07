# You can choose between ['lstm', 'gru', 'lstm_atten']
MODEL = 'lstm'
EPOCHS = 1
LEARNING_RATE = 0.005
# You can choose between ['adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']
OPTIMIZER = 'adam'
ENCODER_OPTIMIZER = 'adam'
DECODER_OPTIMIZER = 'adam'
MAX_SEQ_LEN = 35
EMBED_SIZE = 400
HIDDEN_SIZE = 512
BATCH_SIZE = 64
VOCAB_SIZE = None
LAYERS_COUNT = 1
DROPOUT = float(0.0)
TEACHER_FORCING = 0.5
TRAIN_TEST_SPLIT_RATIO = 0.8
MIN_WORD_COUNT = 2
DATA_PATH = 'data/data.csv'
OUTPUT_PATH = 'results/'
SRC_W2I_FILE = 'src_word_to_index.pkl'
DEST_W2I_FILE = 'dest_word_to_index.pkl'
SRC_I2W_FILE = 'src_index_to_word.pkl'
DEST_I2W_FILE = 'dest_index_to_word.pkl'
SRC_LANG_HEADER = 'Nepali'
DEST_LANG_HEADER = 'English'