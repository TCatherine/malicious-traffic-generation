from parser import parse
from learning import run
import torch
from model import Discriminator, Generator

EPOCHS = 10000
BATCH_SIZE = 512
LEARNING_RATE = 0.00002
BETAS = (0.5, 0.999)
DETERMINATOR_STEP = 100
IMGS_TO_DISPLAY = 100
N_CRITIC = 2
GRADIENT_PENALTY = 10
LOAD_MODEL = False

def is_cuda() -> bool:
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    return torch.cuda.is_available()


def main():
    cuda = is_cuda()
    dataset = parse(
        batch_size=BATCH_SIZE,
        is_cuda=cuda)

    feature_dim = int(list(dataset.sampler.data_source[0][0].shape)[0])
    det_step = DETERMINATOR_STEP * len(dataset)

    D = Discriminator(
        feature_dim=feature_dim,
        lr=LEARNING_RATE,
        betas=BETAS)
    G = Generator(
        feature_dim=feature_dim,
        lr=LEARNING_RATE,
        betas=BETAS)

    if cuda:
        G, D = G.cuda(), D.cuda()

    run(
        data=dataset,
        model=(G, D),
        epochs=EPOCHS,
        det_step=det_step,
        is_cuda=cuda)

if __name__ == "__main__":
    main()
