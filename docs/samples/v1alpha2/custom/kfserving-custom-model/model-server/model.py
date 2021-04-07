import kfserving
from torchvision import models, transforms
from typing import Dict
import torch
from PIL import Image
import base64
import io


class KFServingSampleModel(kfserving.KFModel):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False

    def load(self):
        f = open('imagenet_classes.txt')
        self.classes = [line.strip() for line in f.readlines()]

        model = models.alexnet(pretrained=True)
        model.eval()
        self.model = model

        self.ready = True

    def predict(self, request: Dict) -> Dict:

        return {"predictions": "hello world"}


if __name__ == "__main__":
    model = KFServingSampleModel("kfserving-custom-model")
    model.load()
    kfserving.KFServer(workers=1).start([model])
