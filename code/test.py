import nltk
import torch
import argparse
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
# from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification

from dataset import ABSADataset
from preprocess import *
from utils import *


def get_acccuracy(predictions, labels):
    return torch.sum(labels == predictions)/labels.shape[-1]


def precision_recall_F1(predictions, labels):
    return precision_recall_fscore_support(labels, predictions, average='macro', zero_division=1)


def eval_batch(outputs, labels):
    sftmx = torch.nn.Softmax(dim=2)
    predictions = torch.argmax(sftmx(outputs.logits), dim=2)
    predictions = predictions[torch.where(labels != -100)]
    labels = labels[torch.where(labels != -100)]
    precision, recall, f1, _ = precision_recall_F1(predictions.cpu(), labels.cpu())
    accuracy = get_acccuracy(predictions, labels)
    return precision, recall, f1, accuracy


def test_iter(model, val_loader, device):
    
    total_precision = []
    total_recall = []
    total_f1 = []
    total_accuracy = []

    model.eval()
    
    for batch in tqdm(val_loader):
        labels = batch['labels'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        precision, recall, f1, accuracy = eval_batch(outputs, labels)

        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)
        total_accuracy.append(accuracy)

    print(f'Total precision over this validation set: {sum(total_precision)/len(total_precision)}')
    print(f'Total recall over this validation set: {sum(total_recall)/len(total_recall)}')
    print(f'Total f1 over this validation set: {sum(total_f1)/len(total_f1)}')
    print(f'Total accuracy over this validation set: {sum(total_accuracy)/len(total_accuracy)}')


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_path', type=str, default='code/1764.pt')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--data_type', type=str, default='Restaurant_Dutch')
parser.add_argument('--data_file', type=str, default='data/Test/test_restaurant_dutch.txt')
parser.add_argument('--save_folder', type=str, default='./')
parser.add_argument('--smoothing', type=float, default=0)
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = args.data_file
nltk.download('punkt')

print('Loading Data...')
sentences, aspects, targets = load_data(path)

print('Generating Labels...')
labels, sentences, tag2id, id2tag = generate_labels(sentences, aspects, targets, mode=args.mode, data_type=args.data_type)

print('Encoding Data...')                                                                    
encodings, labels = encode_data(sentences, labels, tag2id, id2tag)

print('Initializing Model...')
# model = XLMRobertaForTokenClassification.from_pretrained('xlm-roberta-base', num_labels=12)
model = torch.load(args.model_path, map_location=torch.device('cpu'))
model.to(device)
model.eval()

print('Initializing Dataset Object...')
dataset = ABSADataset(encodings, labels)

print('Initializing Dataloader')
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

print('Staring Testing...')
with torch.no_grad():
    test_iter(model, loader, device)
        