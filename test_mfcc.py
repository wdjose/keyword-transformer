import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import gc
import argparse

from models import vit
from utils.label_smoothing import LabelSmoothingLoss

parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="Experiment name")
parser.add_argument("--num-heads", default=1, type=int, help="Number of heads in transformer architecture")
parser.add_argument("--no-label-smooth", action="store_false", help="Don't use label smoothing loss")
parser.add_argument("--distill", action="store_true", help="Use distillation token")
parser.add_argument("--version", default=1, type=int, choices=[1, 2], help="Google speech commands version (1 or 2)")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on (cuda vs cpu)")
args = parser.parse_args()

version = int(args.version)                 # version of google-speech-command dataset (v1 or v2)
device = args.device                        # device to run on: GPU or CPU ("cuda" vs "cpu")

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists(os.path.join('results', args.experiment)):
    os.makedirs(os.path.join('results', args.experiment))

MODEL_PATH = os.path.join("models")
MODEL_NAME = "kws-vit"

# data augmentation parameters
time_shift_ms = 100
sample_rate = 16000
time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
num_classes = 11+1


if device == "cuda":
    num_workers = 4
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

heads = args.num_heads
dim = 64*heads
mlp_dim = 256*heads
dim_head = 64
layers = 12
img_x = 98
img_y = 40
patch_x = 1
patch_y = 40

model = vit.ViT(img_x=img_x, img_y=img_y, patch_x=patch_x, 
    patch_y=patch_y, num_classes=num_classes, dim=dim, depth=layers, 
    heads=heads, mlp_dim=mlp_dim, pool='cls', channels=1, 
    dim_head=dim_head, dropout=0., emb_dropout=0.)

print(device)
model.to(device)

loss_fn = F.cross_entropy
if not args.no_label_smooth:
    loss_fn = LabelSmoothingLoss(num_classes, 0.1).to(device)

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


batch_scale = 1
batch_size = 512 // batch_scale
log_interval = 45 * batch_scale

def test(model):
    model.eval()
    correct = 0
    count = 0;
    test_loss = 0.
    with open(os.path.join('data', "v{}_mfcc".format(version), "test.pkl"), 'rb') as f:
        dataset_chunk = pickle.load(f)
        for tuple in dataset_chunk:
            data, target = tuple
            data = torch.from_numpy(data).float().to(device)
            target = torch.from_numpy(target.astype(int)).to(device)

            with torch.no_grad():
                output = model(data)
            test_loss += loss_fn(output.squeeze(), target).item()
            count += 1

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
    
    test_loss /= count
    print(f"\nTest Accuracy: {correct}/{(512*6)} ({100. * correct / (512*6):.1f}%)\tLoss: {test_loss}")
    return correct/(512*6)

model.load_state_dict(torch.load(os.path.join('results', args.experiment, 'best.pth'))['model_state_dict'])
test_acc = test(model)

with open(os.path.join('results', args.experiment, 'test_acc.txt'), 'w') as f:
    f.write('Test Accuracy: {}\n'.format(test_acc))
