import torch
import torchvision


class ResNetHead(torch.nn.Module):
    def __init__(self, encoder, n_feats, n_hidden, n_classes):
        super(ResNetHead, self).__init__()

        self.n_feats = n_feats
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.encoder = encoder
        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_feats, n_hidden),
            torch.nn.ReLU(inplace=True)
        )
        self.clf = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        return self.clf(self.head(self.encoder(x)))

    def encode(self, x):
        return self.head(self.encoder(x))

    def classify(self, z):
        return self.clf(z)


def get_resnet_head(
    resnet_size=50,
    n_hidden=256,
    n_classes=10,
    return_T=False
):
    r"""Auxiliary function for constructing a resnet, given its size.

    Parameters
    ----------
    resnet_size : int, optional (default=50)
        Number of layers in the resnet. Either 18, 34, 50, 101 or 152.
    """
    if resnet_size == 18:
        encoder = torchvision.models.resnet18(weights="IMAGENET1K_V1",
                                              progress=True)
        T = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        n_feats = 512
    elif resnet_size == 34:
        encoder = torchvision.models.resnet34(weights="IMAGENET1K_V1",
                                              progress=True)
        T = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms()
        n_feats = 512
    elif resnet_size == 50:
        encoder = torchvision.models.resnet50(weights="IMAGENET1K_V2",
                                              progress=True)
        T = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        n_feats = 2048
    elif resnet_size == 101:
        encoder = torchvision.models.resnet101(weights="IMAGENET1K_V2",
                                               progress=True)
        T = torchvision.models.ResNet101_Weights.IMAGENET1K_V2.transforms()
        n_feats = 2048
    elif resnet_size == 152:
        encoder = torchvision.models.resnet152(weights="IMAGENET1K_V2",
                                               progress=True)
        T = torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms()
        n_feats = 2048
    else:
        raise ValueError(("Expected resnet_size to be in [18, 34, 50, 101"
                          f", 152], but got {resnet_size}"))
    encoder.fc = torch.nn.Identity()
    model = ResNetHead(encoder,
                       n_feats=n_feats,
                       n_hidden=n_hidden,
                       n_classes=n_classes)

    if return_T:
        return model, T
    else:
        return model
