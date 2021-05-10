import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

from models.kwt import KWT
from datagen import Datagen


if __name__ == '__main__':
    # Python argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--batch-cut", default=1, type=int, help="Cut batch size into multiple model runs if entire batch doesn't fit in VRAM")
    parser.add_argument("--lr", default=0.001, help="Learning rate of AdamW optimizer")
    parser.add_argument("--weight-decay", default=0.1, help="Weight decay of AdamW optimizer")
    parser.add_argument("--no-cosine", action='store_false', help="Indicates whether cosine annealing lr scheduler is used (default=no)")
    parser.add_argument("--warmup-epochs", default=10, type=int, help="Number of epochs for learning rate warmup")
    parser.add_argument("--num-heads", default=1, type=int, help="Number of heads in transformer architecture")
    parser.add_argument("--no-label-smooth", action="store_false", help="Don't use label smoothing loss")
    parser.add_argument("--distill", action="store_true", help="Use distillation token")
    parser.add_argument("--num-steps", default=23000, type=int, help="Number of training steps")
    parser.add_argument("--version", default=1, type=int, choices=[1, 2, 3], help="Google speech commands version (1 or 2)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on (cuda vs cpu)")
    parser.add_argument("--num-workers", default=10, type=int, help="Number of workers for data augmentation")
    args = parser.parse_args()

    print("Running: python {}".format(sys.argv))


    batch_size = args.batch_size                # maximum of batch_size=512
    batch_cut = args.batch_cut                  # if entire batch doesn't fit in VRAM, divide into smaller batches
    lr = float(args.lr)                         # learning rate of AdamW optimizer
    weight_decay = float(args.weight_decay)     # weight decay of AdamW optimizer
    steps_per_epoch = 45*(512//batch_size)      # how many steps per epoch
    warmup_epochs = args.warmup_epochs          # number of warmup epochs
    use_cosine_lr = args.no_cosine              # boolean variable indicating if cosine annealing lr scheduler is used or not
    num_steps = int(args.num_steps)             # number of training steps (default: 23,000 steps)
    epochs = 512                                # number of epochs --> used only for lr scheduler; this is fixed, edit num_steps instead
    version = args.version                      # version of google-speech-command dataset (v1 or v2)
    device = args.device                        # device to run on: GPU or CPU ("cuda" vs "cpu")

    if device == "cuda":
        num_workers = int(args.num_workers)
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(os.path.join('results', args.experiment)):
        os.makedirs(os.path.join('results', args.experiment))

    MODEL_PATH = os.path.join("models")
    MODEL_NAME = "kwt"

    # data augmentation parameters
    time_shift_ms = 100
    sample_rate = 16000
    time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
    num_classes = 12 if version != 3 else 35
    print("num_classes:", num_classes, type(version))

    # model parameters
    heads = args.num_heads
    dim = 64*heads
    mlp_dim = 256*heads
    dim_head = 64
    layers = 12
    img_x = 98
    img_y = 40
    patch_x = 1
    patch_y = 40

    # initialize model
    model = KWT(img_x=img_x, img_y=img_y, patch_x=patch_x, 
        patch_y=patch_y, num_classes=num_classes, dim=dim, depth=layers, 
        heads=heads, mlp_dim=mlp_dim, pool='cls', channels=1, 
        dim_head=dim_head, dropout=0., emb_dropout=0.)
    model.to(device)

    # dataset and dataloader initialization
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, part="train", steps=None, batch_size=512, 
            version="1", preprocess="mfcc"):
                super(Dataset).__init__()
                self.dataset = None
                self.datasetworkers = dict()
                self.batch_size = batch_size
                self.version = version
                self.preprocess = preprocess
                self.steps = steps
                self.part = part
        def __len__(self):
                if self.steps is not None:
                    return self.steps
                worker_info = torch.utils.data.get_worker_info()
                if self.dataset is None:
                    self.dataset = Datagen(
                        batch_size=self.batch_size, 
                        version=self.version, 
                        preprocess=self.preprocess)
                if worker_info is not None:
                    return self.datasetworkers[worker_info.id].dataLen(self.part)
                return self.dataset.dataLen(self.part)
        def __getitem__(self, index):
                worker_info = torch.utils.data.get_worker_info()
                dataset = self.dataset if worker_info is None else self.datasetworkers[worker_info.id]
                data, target = dataset.getData(self.part, index)
                target = target.astype(int)
                return data, target

    test_set = Dataset(part="test", batch_size=1, version=version, preprocess="mfcc")
    test_generator = torch.utils.data.DataLoader(test_set, num_workers=0, batch_size=None)
    
    def number_of_correct(pred, target):
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(tensor):
        return tensor.argmax(dim=-1)

    # testing function
    def test(model):
        model.eval()
        correct = 0
        count = 0;
        for data, target in tqdm(test_generator):
            data = data.float().to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(data)
            count += 1

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
        
        print(f"Test Accuracy: {correct}/{(count)} ({100. * correct / (count):.2f}%)")
        return correct/(count)

    model.load_state_dict(torch.load(os.path.join('results', args.experiment, 'best.pth'))['model_state_dict'])
    test_acc = test(model)

    with open(os.path.join('results', args.experiment, 'test_acc.txt'), 'w') as f:
        f.write('Test Accuracy: {}\n'.format(test_acc))
