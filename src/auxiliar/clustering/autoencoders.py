from autoencoder import Decoder, Encoder, LitAutoEncoder
from deep_autoencoder import Autoencoder


def new_one_band_autoencoder(width:int, height:int):
    return LitAutoEncoder(Encoder(width*height), Decoder(width*height))

def new_n_band_deep_autoencoder(base_channel:int, latent_dim:int, num_input_channels: int = 3,width: int = 32,height: int = 32):
    return Autoencoder(base_channel,
                 latent_dim,
                 num_input_channels=num_input_channels,
                 width=width,
                 height=height)