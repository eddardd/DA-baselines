import torch
import torchvision
from torchvision.models import ViT_B_16_Weights
from torchvision.models import ViT_B_32_Weights
from torchvision.models import ViT_L_16_Weights
from torchvision.models import ViT_L_32_Weights
from torchvision.models import ViT_H_14_Weights


class VisionTransformer(torch.nn.Module):
    def __init__(self, encoder, n_feats, n_classes):
        super(VisionTransformer, self).__init__()

        self.n_feats = n_feats
        self.n_classes = n_classes
        self.encoder = encoder
        self.clf = torch.nn.Linear(n_feats, n_classes)

    def forward(self, x):
        return self.clf(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def classify(self, z):
        return self.clf(z)


def get_vit(n_layers=14,
            name="b",
            n_classes=10,
            return_T=False):
    r"""Auxiliary function for constructing a VisionTransformer, given its
    number of layers and type.

    Parameters
    ----------
    resnet_size : int, optional (default=50)
        Number of layers in the resnet. Either 18, 34, 50, 101 or 152.
    """
    if n_layers == 16 and name.lower() == 'b':
        encoder = torchvision.models.vit_b_16(
            weights="IMAGENET1K_V1", progress=True)
        T = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        n_feats = 768
    elif n_layers == 32 and name.lower() == 'b':
        encoder = torchvision.models.vit_b_32(
            weights="IMAGENET1K_V1", progress=True)
        T = ViT_B_32_Weights.IMAGENET1K_V1.transforms()
        n_feats = 768
    elif n_layers == 16 and name.lower() == 'l':
        encoder = torchvision.models.vit_l_16(
            weights="IMAGENET1K_V1", progress=True)
        T = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
        n_feats = 1024
    elif n_layers == 32 and name.lower() == 'l':
        encoder = torchvision.models.vit_l_32(
            weights="IMAGENET1K_V1", progress=True)
        T = ViT_L_32_Weights.IMAGENET1K_V1.transforms()
        n_feats = 1024
    elif n_layers == 14 and name.lower() == 'h':
        encoder = torchvision.models.vit_h_14(
            weights="IMAGENET1K_SWAG_E2E_V1", progress=True)
        T = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        n_feats = 1280
    else:
        raise ValueError(("Expected (n_layers, name) to be valid, "
                          f"but got {n_layers, name}"))
    encoder.heads = torch.nn.Identity()
    model = VisionTransformer(encoder, n_feats, n_classes)

    if return_T:
        return model, T
    else:
        return model
