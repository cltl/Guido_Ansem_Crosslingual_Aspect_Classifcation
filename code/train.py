import torch
# import wandb
from tqdm import tqdm
from utils import calculate_loss


# def train_log(loss, example_ct):
#     wandb.log({ "loss": loss}, step=example_ct)
#     print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def training_iter(model, optim, train_loader, critereon, device, n_batch, t_writer):
    losses = []
    model.train()
    # wandb.watch(model, critereon, log="all", log_freq=25)
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = calculate_loss(outputs, labels, critereon, device)
        loss.backward()
        optim.step()

        # if (n_batch + 1) % 25 == 0:
        #     train_log(loss, n_batch)

        t_writer.add_scalar('loss', loss, n_batch)

        losses.append(loss)
        n_batch += 1
        
    print(f'Average loss over this epoch: {sum(losses)/len(losses)}')
    return model