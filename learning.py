from torch.autograd import Variable
import numpy as np
import torch

def compute_gradient_penalty(
        D,
        real_samples,
        fake_samples,
        Tensor
    ):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def run(
        data,
        model,
        epochs,
        det_step,
        is_cuda=False
        ):

    G, D = model
    feature_dim = int(list(data.sampler.data_source[0][0].shape)[0])

    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    # Load previous model 
    # DB = 'generator'
    # model_path = os.path.join('./model', )
    # if LOAD_MODEL:
    #     G.load_state_dict(torch.load(os.path.join(model_path, 'generator.pkl')))
    #     D.load_state_dict(torch.load(os.path.join(model_path, 'critic.pkl')))

    step = 0
    for epoch in range(epochs):
        for i, (batch, _) in enumerate(data):
            real_data = Variable(batch.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (batch.shape[0], feature_dim))))
            # Generate a batch of images
            fake_data = G(z)

            # Real and fake data
            real_validity = D(real_data)
            fake_validity = D(fake_data)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                D,
                real_data.data,
                fake_data.data,
                Tensor
            )
            # Adversarial loss
            lambda_gp = 0.2

            d_loss = - torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            D.optimizer.zero_grad()
            d_loss.backward()
            D.optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            if step % det_step == 0:

                fake_data = G(z)

                fake_validity = D(fake_data)
                g_loss = - torch.mean(fake_validity)

                G.optimizer.zero_grad()
                g_loss.backward()
                G.optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f (RV: %f)(FV: %f)] [G loss: %f] [Penalty: %f]"
                    % (epoch, epochs, i, len(data), d_loss.item(), torch.mean(real_validity), torch.mean(fake_validity), g_loss.item(), gradient_penalty.item())
                )

            step+=1

            # if batches_done % opt.sample_interval == 0:
            #     save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            # batches_done += opt.n_critic