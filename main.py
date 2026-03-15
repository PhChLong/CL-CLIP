from models import CLIPWrapper
from data import get_task_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from PIL import Image

# tasks = get_task_sequence()
# print(tasks)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPWrapper(device=device)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = ['a photo of a cat', 'a photo of a dog']
outputs = clip(text, image)

logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

print(probs)