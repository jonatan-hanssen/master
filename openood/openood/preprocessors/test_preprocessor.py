import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .base_preprocessor import BasePreprocessor
from .transform import Convert, Mask


class TestStandardPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""

    def __init__(self, config: Config):
        super(TestStandardPreProcessor, self).__init__(config)
        if 'hyperkvasir' in config.dataset.name:
            self.transform = tvs_trans.Compose(
                [
                    Convert('RGB'),
                    tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
                    tvs_trans.CenterCrop(self.image_size),
                    Mask(),
                    tvs_trans.ToTensor(),
                    tvs_trans.Normalize(mean=self.mean, std=self.std),
                ]
            )
        else:
            self.transform = tvs_trans.Compose(
                [
                    Convert('RGB'),
                    tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
                    tvs_trans.CenterCrop(self.image_size),
                    tvs_trans.ToTensor(),
                    tvs_trans.Normalize(mean=self.mean, std=self.std),
                ]
            )
