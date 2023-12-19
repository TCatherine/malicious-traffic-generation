from pathlib import Path

dict_path = Path('data') / 'dict.model'

EPOCHS = 10000
BATCH_SIZE = 48
LEARNING_RATE = 0.00002
BETAS = (0.5, 0.999)
DETERMINATOR_STEP = 100
IMGS_TO_DISPLAY = 100
N_CRITIC = 2
GRADIENT_PENALTY = 10
LOAD_MODEL = False

BPE = {
    "vocab_size": 1000,
    "fixed_length": 500,
    "model_path": dict_path,
}
