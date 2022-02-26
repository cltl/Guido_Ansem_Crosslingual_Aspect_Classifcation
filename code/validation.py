import torch
from tqdm import tqdm
from numpy import average
from sklearn.metrics import precision_recall_fscore_support

from utils import calculate_loss

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


def validation_iter(model, val_loader, device, critereon, n_batch, v_writer, data_type):
    print('Starting Validation...')
    total_loss = []
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
        loss = calculate_loss(outputs, labels, critereon, device)

        total_loss.append(loss)
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)
        total_accuracy.append(accuracy)

    torch.save(model, f'model{data_type}_{n_batch}.pt')

    v_writer.add_scalar('loss', sum(total_loss)/len(total_loss), n_batch)
    v_writer.add_scalar('Precision', sum(total_precision)/len(total_precision), n_batch)
    v_writer.add_scalar('Recall', sum(total_recall)/len(total_recall), n_batch)
    v_writer.add_scalar('F1', sum(total_f1)/len(total_f1), n_batch)
    v_writer.add_scalar('Accuracy', sum(total_accuracy)/len(total_accuracy), n_batch)

    print(f'Average loss over this validation set: {sum(total_loss)/len(total_loss)}')
    print(f'Total precision over this validation set: {sum(total_precision)/len(total_precision)}')
    print(f'Total recall over this validation set: {sum(total_recall)/len(total_recall)}')
    print(f'Total f1 over this validation set: {sum(total_f1)/len(total_f1)}')
    print(f'Total accuracy over this validation set: {sum(total_accuracy)/len(total_accuracy)}')

    model.train()
    return model