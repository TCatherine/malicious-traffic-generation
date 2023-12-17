from pathlib import Path

dict_path = Path(__file__).parent / 'parser' / 'penguin_of_doom.vocab'

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
    "pairable_chars": "a-zA-Z0-9",
    "dict_path": dict_path,
    "fixed_length": 500
}
