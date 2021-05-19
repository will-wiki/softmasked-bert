
import torch
import pandas
import torch.nn as nn
from data import Data
from config import Config
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_data import CSC_DataSet, collate_fn
from model.SoftMasked_Bert import SoftMasked_Bert

vocab_file = '../pretrain_model/chinese_wwm_ext_pytorch/vocab.txt'
config = Config()

train_dataset = CSC_DataSet('./data/narts/SIGHAN15_train.csv', config.vocab_file)
test_dataset = CSC_DataSet('./data/narts/SIGHAN15_test.csv', config.vocab_file)
train_generator = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
test_generator = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
model = SoftMasked_Bert(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = Adam(model.parameters(), lr=config.lr)
criterion_n, criterion_b = nn.NLLLoss(), nn.BCELoss()
gama = 0.8


def dev(dataloader, model):
    model.eval()
    avg_loss, total_element = 0, 0
    d_correct, c_correct = 0, 0
    for i, batch_data in enumerate(dataloader):
        batch_input_ids, batch_input_mask, \
        batch_segment_ids, batch_output_ids, batch_labels = batch_data
        batch_input_ids = batch_input_ids.to(device)
        batch_input_mask = batch_input_mask.to(device)
        batch_segment_ids = batch_segment_ids.to(device)
        batch_output_ids = batch_output_ids.to(device)
        batch_labels = batch_labels.to(device)
        output, prob = model(batch_input_ids, batch_input_mask, batch_segment_ids)
        # correct = out.argmax(dim=-1).eq(data["output_ids"]).sum().item()
        output = output.argmax(dim=-1)
        c_correct += sum([output[i].equal(batch_output_ids[i]) for i in range(len(output))])
        prob = torch.round(prob).long()
        d_correct += sum([prob[i].squeeze().equal(batch_labels[i]) for i in range(len(prob))])

        total_element += len(batch_data)

    print("d_acc=", d_correct / total_element, "c_acc", c_correct / total_element)

for epoch in range(config.epoch):

    model.train()
    avg_loss, total_element = 0, 0
    d_correct, c_correct = 0, 0
    for i, batch_data in enumerate(train_generator):
        batch_input_ids, batch_input_mask, \
        batch_segment_ids, batch_output_ids, batch_labels = batch_data
        batch_input_ids = batch_input_ids.to(device)
        batch_input_mask = batch_input_mask.to(device)
        batch_segment_ids = batch_segment_ids.to(device)
        batch_output_ids = batch_output_ids.to(device)
        batch_labels = batch_labels.to(device)
        output, prob = model(batch_input_ids, batch_input_mask, batch_segment_ids)
        loss_b = criterion_b(prob, batch_labels.float())
        loss_n = criterion_n(output.reshape(-1, output.size()[-1]), \
                             batch_output_ids.reshape(-1))
        loss = gama * loss_n + (1 - gama) * loss_b
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # correct = out.argmax(dim=-1).eq(data["output_ids"]).sum().item()
        output = output.argmax(dim=-1)
        c_correct += sum([output[i].equal(batch_output_ids[i]) for i in range(len(output))])
        prob = torch.round(prob).long()
        opp = prob[0].squeeze()
        opp1 = batch_labels[0]
        d_correct += sum([prob[i].squeeze().equal(batch_labels[i]) for i in range(len(prob))])

        avg_loss += loss.item()
        #     total_correct += c_correct
        #     # total_element += data["label"].nelement()
        total_element += len(batch_data)

    post_fix = {
        "epoch": epoch,
        "iter": i,
        "avg_loss": avg_loss / (i + 1),
        "d_acc": d_correct / total_element,
        "c_acc": c_correct / total_element
    }

    # if i % self.log_freq == 0:
    #     data_loader.write(str(post_fix))

    print("EP%d_, avg_loss=" % (epoch), avg_loss / len(train_generator), "d_acc=",
          d_correct / total_element, "c_acc", c_correct / total_element)
    dev(test_generator, model)
