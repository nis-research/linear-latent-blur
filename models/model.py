import os
import argparse
import torch.nn
import torch.nn as nn
from config import NORMALIZE_IMAGES, USE_TEN_CROP, INPUT_CHANNELS, MAX_EPOCHS, TRANSITION_LENGTH, LEARNING_RATE, \
    LAYERS_DEPTH, BATCH_SIZE
from utils import prepare_batch, denormalize_image, init_logger
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image
from dataset import datasetModules as ds
from visualizations import create_transition_directories
# from torch.utils.tensorboard import SummaryWriter


class AE(pl.LightningModule):
    """
    The architecture is a slightly modified version of the model at
    https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py, distributed through the Apache License 2.0.
    """

    def __init__(self, experiment_name="", in_channels=INPUT_CHANNELS, hidden_dims=None, model_type="baseline",
                 pretrain=False) -> None:

        super().__init__()
        self.model_type = model_type
        self.pretrain = pretrain
        # os.makedirs(os.path.join(f"tensorboardLogger/{experiment_name}"), exist_ok=True)
        # self.writer = SummaryWriter(os.path.join(f"tensorboardLogger/{experiment_name}"))
        self.epoch_idx = 0

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 1024]

        # encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),  # double # of channels, half dimensions
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # decoder
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=INPUT_CHANNELS,
                      kernel_size=3, padding=1),
            nn.Tanh() if NORMALIZE_IMAGES else nn.Sigmoid())

    def set_model_type(self, model_type):
        self.model_type = model_type

    def encode_decode(self, input):
        """
        Returns the encoded input and its decoded version.
        """
        down = self.encoder(input)
        return down, self.final_layer(self.decoder(down))

    def decode(self, input):
        return self.final_layer(self.decoder(input))

    def _get_loss(self, batch):
        """
        Given a batch of images, this function returns the loss function, based on the model type:
        - if in `pretrain` mode, the loss is L1
        - if in `baseline` mode, the loss if mean of the L1 loss for both of the input images
        - if in `indirect` mode the loss is the `baseline` loss + an interpolation L1 loss
        - if in `direct` mode, the loss in the `indirect` loss + an latent representation loss
        """
        if self.pretrain:
            imgs = prepare_batch(batch, pretrain=True)
            enc, rec = self.encode_decode(imgs)
            return F.mse_loss(rec, imgs)

        img1, img2, target = prepare_batch(batch)
        img1_enc, img1_hat = self.encode_decode(img1)
        img2_enc, img2_hat = self.encode_decode(img2)
        loss = 1 / 2 * (F.mse_loss(img1_hat, img1) + F.mse_loss(img2_hat, img2))
        if self.model_type == "baseline":
            return loss

        interp_repr = 0.5 * img1_enc + 0.5 * img2_enc
        interp_dec = self.decode(interp_repr)
        interp_loss = F.mse_loss(interp_dec, target)

        if self.model_type == "indirect":
            return loss + interp_loss
        elif "direct" in self.model_type:
            target_enc, _ = self.encode_decode(target)
            return loss + interp_loss + F.mse_loss(interp_repr, target_enc)
        raise Exception("Unknown model type")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('val_loss', loss)
        return {'val_loss': loss.detach()}

    def training_epoch_end(self, outputs) -> None:
        batch_losses = [x["loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        print(f"Train epoch loss - {epoch_loss.item()}")

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        print(f"Validation epoch loss - {epoch_loss.item()}")
        return {'epoch_val_loss': epoch_loss.item()}

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log('test_loss', loss)
        return loss.detach()


class TransitionCallback(pl.Callback):
    """
    Callback used to generate a transition from z-stack 0 to z-stack 16 of reconstructed images from interpolated latent
    representations.
    """

    def __init__(self, input_left, input_right, target, steps, experiment_name, slide_type, every_n_epochs=10):
        self.step = 1 / (steps + 1)  # by how much the interpolation parameter increases after each interpolation step
        self.input_left = input_left
        self.input_right = input_right
        self.every_n_epochs = every_n_epochs
        self.target = target
        self.experiment_name = experiment_name
        self.save_path = os.path.join(f"{slide_type}-experiments", experiment_name)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            print("Generating transitions...")
            input_left = self.input_left.to(pl_module.device)
            input_right = self.input_right.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                # Reconstruct images
                enc_left, dec_left = pl_module.encode_decode(input_left)
                enc_right, dec_right = pl_module.encode_decode(input_right)
                alpha = self.step
                for i, img in enumerate(dec_left):
                    img = denormalize_image(img, self.experiment_name) if NORMALIZE_IMAGES else img
                    save_image(img, os.path.join(self.save_path, f"transition-{self.experiment_name}-{i}",
                                                 f"z0_epoch_{trainer.current_epoch}.png"))
                while alpha < 1:
                    interp = (1 - alpha) * (enc_left) + alpha * (enc_right)
                    interp_dec = pl_module.decode(interp)
                    for i, img in enumerate(interp_dec):
                        img = denormalize_image(img, self.experiment_name) if NORMALIZE_IMAGES else img
                        save_image(img, os.path.join(self.save_path, f"transition-{self.experiment_name}-{i}",
                                                     f"z{int(16 * alpha)}_epoch_{trainer.current_epoch}.png"))
                    alpha += self.step
                pl_module.train()
            for i, img in enumerate(dec_right):
                img = denormalize_image(img, self.experiment_name) if NORMALIZE_IMAGES else img
                save_image(img, os.path.join(self.save_path, f"transition-{self.experiment_name}-{i}",
                                             f"z16_epoch_{trainer.current_epoch}.png"))


class SaveEncodingsCallback(pl.Callback):
    """
    Callback used to save the latent representations for some test images, once every `n` epochs.
    """

    def __init__(self, images, every_n_epochs, experiment_name):
        super(SaveEncodingsCallback, self).__init__()
        self.images = images
        self.every_n_epochs = every_n_epochs
        self.exp_name = experiment_name

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            print("Saving encodings...")
            images = self.images.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                encodings, _ = pl_module.encode_decode(images)
                torch.save(encodings,
                           os.path.join(f"encodings-{self.exp_name}", f"encodings_epoch{trainer.current_epoch}.pt"))
                pl_module.train()


def train(experiment_name=None, hidden_dims=None, model_type="baseline", slide_type=None, input_path="", pretrain=False):
    assert model_type in ["baseline", "indirect", "direct"]
    assert slide_type in ["w1", "w2"]
    pl.seed_everything(42)
    logger = init_logger(experiment_name)
    print(experiment_name)
    print(slide_type)
    try:
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"Device: {device}")
        os.makedirs("checkpoints", exist_ok=True)
        if pretrain:  # used to train a basic AE for input reconstruction, to check network architecture corectness
            unet = AE(experiment_name, INPUT_CHANNELS, hidden_dims, model_type, True)
            datamodule = ds.PretrainDataModule(batch_size=2, slide_type=slide_type)
        else:
            unet = AE(experiment_name, INPUT_CHANNELS, hidden_dims, model_type)
            datamodule = ds.TrainDataModule(batch_size=BATCH_SIZE, slide_type=slide_type, normalize=NORMALIZE_IMAGES,
                                            use_ten_crop=USE_TEN_CROP, input_path=input_path)

        images = ds.input_for_visualization(slide_type, input_path=input_path)
        # create a directory with a transition from z0 to z16 of the original images
        create_transition_directories(experiment_name, images, slide_type)

        # train locally on MAC with GPU ("mps')
        # trainer = pl.Trainer(deterministic=True, max_epochs=MAX_EPOCHS, log_every_n_steps=10, accumulate_grad_batches=16,
        #                      accelerator="mps", devices=1, precision=16, default_root_dir=os.path.join("checkpoints"),
        #                      callbacks=[TransitionCallback(input_left=images[0], input_right=images[-1], target=images[2],
        #                                                    steps=TRANSITION_LENGTH - 2, experiment_name=experiment_name,
        #                                                    slide_type=slide_type)])

        # train on Snellius server with 1 GPU
        trainer = pl.Trainer(deterministic=True, max_epochs=MAX_EPOCHS, log_every_n_steps=10, accumulate_grad_batches=16,
                             accelerator="gpu", gpus=1, devices=1, precision=16, default_root_dir=os.path.join("checkpoints"),
                             callbacks=[TransitionCallback(input_left=images[0], input_right=images[-1], target=images[2],
                                                           steps=TRANSITION_LENGTH - 2, experiment_name=experiment_name,
                                                           slide_type=slide_type)])

        if torch.cuda.is_available():
            unet = unet.to(device)
        trainer.fit(unet, datamodule=datamodule)
    except:
        logger.error("Error: \n", exc_info=True)
        raise

    # flush and close logs writer
    # unet.writer.flush()
    # unet.writer.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-experiment", required=True, help="The name of the experiment.")
    parser.add_argument("--model-type", required=False, choices=["baseline", "indirect", "direct"], default="baseline",
                        help="The model type describes the type of loss that will be used.")
    parser.add_argument("--slide-type", required=False, choices=["w1", "w2"], default="w1",
                        help="With which type of slides (w1 or w2) will the model be trained?")
    parser.add_argument("--input-path", required=False, help="Path to directory where train/test/val sets are located."
                                                             "Only needed if the images are not in the `/dataset` "
                                                             "directory of the project.")

    args = parser.parse_args()

    input_path = args.input_path
    slide_type = args.slide_type
    exp_name = args.experiment
    model_type = args.model_type

    train(exp_name, LAYERS_DEPTH, model_type, slide_type, input_path)

    # # Remember to write down for each experiment the vnum, last epoch and last step (needed for the testing part)
    # train("w2-baseline", LAYERS_DEPTH, "baseline", "w2", args.input_path)
    # train("w2-indirect", LAYERS_DEPTH, "indirect", "w2", args.input_path)
    # train("w2-direct", LAYERS_DEPTH, "direct", "w2", args.input_path)

    # train("w1-baseline", LAYERS_DEPTH, "baseline", "w1", args.input_path)
    # train("w1-indirect", LAYERS_DEPTH, "indirect", "w1", args.input_path)
    # train("w1-direct", LAYERS_DEPTH, "direct", "w1", args.input_path)
