import torch
from PIL import Image
import argparse

from transformers import CLIPProcessor, CLIPModel


parser = argparse.ArgumentParser()
parser.add_argument("--data_url", default=None)
parser.add_argument("--vision_tower", default="openai/clip-vit-base-patch14")
args = parser.parse_args()


class CLIPVisionTower():
    def __init__(self, args):

        self.is_loaded = False
        self.vision_tower_name = args.vision_tower

    def load_model(self):

        self.processor = CLIPProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPModel.from_pretrained(self.vision_tower_name)

    @torch.inference_mode
    def forward(self, image, text):
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = self.vision_tower(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)


_encoder = CLIPVisionTower()


def use_encoder():
    with open(args.data_url) as data:
        for _values in data.values():
            image = _values['image']
            text = _values['text']

    if _encoder.is_loaded:
        _encoder.forward(image=image, text=text)
