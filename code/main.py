import os
import nltk
import copy
import torch
# import wandb
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification

from validation import validation_iter
from train import training_iter
from dataset import ABSADataset
from preprocess import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--smoothing', type=float, default=0)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=5)

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_type', type=str, default='Restaurant_Dutch')
parser.add_argument('--data_file', type=str, default='data/Train/train_camera_german.txt')
parser.add_argument('--save_folder', type=str, default='./')
args = parser.parse_args()

t_writer = SummaryWriter(os.path.join(args.save_folder, f'train/{args.lr}-{args.smoothing}'), flush_secs=5)
v_writer = SummaryWriter(os.path.join(args.save_folder, f'val/{args.lr}-{args.smoothing}'), flush_secs=5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = args.data_file
nltk.download('punkt')

print('Loading Data...')
sentences, aspects, targets = load_data(path)

print('Generating Labels...')
labels, sentences, tag2id, id2tag = generate_labels(sentences, aspects, targets, mode=args.mode, data_type=args.data_type)
train_sents, val_sents, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, 
                                                                    shuffle=True, random_state=666)

print('Encoding Data...')                                                                    
train_encodings, train_labels = encode_data(train_sents, train_labels, tag2id, id2tag)
val_encodings, val_labels = encode_data(val_sents, val_labels, tag2id, id2tag)

print('Initializing Model...')
num_labels = len(set(itertools.chain(*labels)))
model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
model.to(device)
model.train()

print('Initializing Dataset Object...')
train_dataset = ABSADataset(train_encodings, train_labels)
validation_dataset = ABSADataset(val_encodings, val_labels)

print('Initializing Dataloader & Optimizer')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
optim = Adam(model.parameters(), lr=args.lr)
critereon = init_loss_function(labels, args.smoothing)

print('Staring Training...')
n_batch = 0
for epoch in range(args.epochs):
    print(f'Starting Epoch {epoch + 1}...')
    model = training_iter(model, optim, train_loader, critereon, device, n_batch, t_writer)
    n_batch += len(train_loader)
    with torch.no_grad():
        model = validation_iter(model, validation_loader, device, critereon, n_batch, v_writer, args.data_type)
        