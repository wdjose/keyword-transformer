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
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.label_smoothing import LabelSmoothingLoss


parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="Experiment name")
parser.add_argument("--batch-size", default=512, type=int)
parser.add_argument("--batch-cut", default=1, type=int, help="Cut batch size into multiple model runs if entire batch doesn't fit in VRAM")
parser.add_argument("--no-cosine", action='store_false', help="Indicates whether cosine annealing lr scheduler is used (default=no)")
parser.add_argument("--warmup-epochs", default=10, type=int, help="Number of epochs for learning rate warmup")
parser.add_argument("--num-heads", default=1, type=int, help="Number of heads in transformer architecture")
parser.add_argument("--no-label-smooth", action="store_false", help="Don't use label smoothing loss")
parser.add_argument("--distill", action="store_true", help="Use distillation token")
parser.add_argument("--version", default=1, type=int, choices=[1, 2], help="Google speech commands version (1 or 2)")
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on (cuda vs cpu)")
args = parser.parse_args()

print("Running: python {}".format(sys.argv))


batch_size = args.batch_size                # maximum of batch_size=512
batch_cut = args.batch_cut                  # if entire batch doesn't fit in VRAM, divide into smaller batches
steps_per_epoch = 45*(512//batch_size)      # how many steps per epoch
epochs = 512                                # number of epochs (fixed to 512 due to stored data augmentation)
warmup_epochs = args.warmup_epochs          # number of warmup epochs
use_cosine_lr = args.no_cosine              # boolean variable indicating if cosine annealing lr scheduler is used or not
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

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

n = count_parameters(model)
print("Number of parameters: %s" % n)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warmup_epochs, verbose=True) if use_cosine_lr else None
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)


loss_fn = F.cross_entropy
if not args.no_label_smooth:
    loss_fn = LabelSmoothingLoss(num_classes, 0.1).to(device)


best_val_acc = 0.
def val(model, epoch, logfile=None):
    model.eval()
    correct = 0
    count = 0;
    val_loss = 0.
    with open(os.path.join('data', "v{}_mfcc".format(version), "val.pkl"), 'rb') as f:
        dataset_chunk = pickle.load(f)
        for tuple in dataset_chunk:
            data, target = tuple
            data = torch.from_numpy(data).float().to(device)
            target = torch.from_numpy(target.astype(int)).to(device)

            with torch.no_grad():
                output = model(data)
            val_loss += loss_fn(output.squeeze(), target).item()
            count += 1

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
    
    val_loss /= count
    val_acc.append(correct/(512*6))
    val_losses.append(val_loss)
    print(f"Epoch: {epoch}\tAccuracy: {correct}/{(512*6)} ({100. * correct / (512*6):.1f}%)\tLoss: {train_losses[-1]} {val_loss}")
    if logfile is not None:
        logfile.write('{},{},{},{}\n'.format(epoch, train_losses[-1], val_losses[-1], val_acc[-1]))
        logfile.flush()
    global best_val_acc
    if ((correct/(512*6)) > best_val_acc):
        best_val_acc = correct/(512*6)
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'val_acc': best_val_acc
            },os.path.join('results', args.experiment, 'best.pth'))

def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


logfile = open(os.path.join('results', args.experiment, 'data.txt'), 'w')
logfile.write('Epoch,Train Losses,Val Losses,Val Accuracy\n')

# scaler = torch.cuda.amp.GradScaler()
model.train()
training_step = 0
train_losses = []
val_losses = []
val_acc = []
for file_id in tqdm(range(1, 61)):
    with open(os.path.join('data', "v{}_mfcc".format(version), "{}.pkl".format(file_id)), 'rb') as f:
        dataset_chunk = pickle.load(f)
        for tuple in dataset_chunk:
            num_batches = len(tuple[0]) // batch_size
            for batchnum in range(num_batches):
                optimizer.zero_grad()
                for batchpart in range(batch_cut):
                    start_idx = batchnum*batch_size + batchpart*(batch_size//batch_cut)
                    end_idx = batchnum*batch_size + (batchpart+1)*(batch_size//batch_cut)
                    data = tuple[0][start_idx : end_idx]
                    target = tuple[1][start_idx : end_idx]
                    
                    data = torch.from_numpy(data).float().to(device)
                    target = torch.from_numpy(target.astype(int)).to(device)
            
                    # with torch.cuda.amp.autocast():
                    output = model(data)
                    # assert output.dtype is torch.float16
                    loss = loss_fn(output.squeeze(), target)
                    assert loss.dtype is torch.float32

                    loss.backward()

                optimizer.step()

                training_step += 1
                train_losses.append(loss.item())
                if training_step % steps_per_epoch == 0:
                    val(model, training_step//steps_per_epoch, logfile)
                    model.train()
                    scheduler.step()
    del dataset_chunk
    gc.collect()

logfile.close()

torch.save({
    'epoch': training_step//steps_per_epoch, 
    'model_state_dict': model.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'val_acc': val_acc[-1]
    },os.path.join('results', args.experiment, 'last.pth'))

def test(model):
    model.eval()
    correct = 0
    count = 0;
    with open(os.path.join('data', "v{}_mfcc".format(version), "test.pkl"), 'rb') as f:
        dataset_chunk = pickle.load(f)
        for tuple in dataset_chunk:
            data, target = tuple
            data = torch.from_numpy(data).float().to(device)
            target = torch.from_numpy(target.astype(int)).to(device)

            with torch.no_grad():
                output = model(data)
            count += target.shape[-1]

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
    
    print(f"Test Accuracy: {correct}/{(count)} ({100. * correct / (count):.1f}%)")
    return correct/(count)

model.load_state_dict(torch.load(os.path.join('results', args.experiment, 'best.pth'))['model_state_dict'])
test_acc = test(model)

with open(os.path.join('results', args.experiment, 'test_acc.txt'), 'w') as f:
    f.write('Test Accuracy: {}\n'.format(test_acc))
