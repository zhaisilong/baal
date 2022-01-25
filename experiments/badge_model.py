import os

import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.vgg import make_layers, model_urls


pjoin = os.path.join
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        'M',
    ],
    'E': [
        64,
        64,
        'M',
        128,
        128,
        'M',
        256,
        256,
        256,
        256,
        'M',
        512,
        512,
        512,
        512,
        'M',
        512,
        512,
        512,
        512,
        'M',
    ],
}


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )
        self.after_embedding = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, num_classes),)
        if init_weights:
            self._initialize_weights()
        self.e1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.e1 = self.classifier(x)
        x = self.after_embedding(self.e1)
        return x

    def get_embeddings(self):
        return self.e1

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False

    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model_dir = '/app/model' if os.path.exists('/app/model') else None
        d = model_zoo.load_url(model_urls['vgg16'], model_dir=model_dir)
        if kwargs['num_classes'] != 1000:
            d = {k: v for k, v in d.items() if ('classifier' not in k) and ('after_embedding' not in k)}
        model.load_state_dict(d, strict=False)
    return model
