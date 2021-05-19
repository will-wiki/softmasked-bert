
import pandas
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CSC_DataSet(Dataset):

    def __init__(self, file_path, vocab_file):
        super(CSC_DataSet, self).__init__()
        self.datas = pandas.read_csv(file_path)
        self.size = len(self.datas)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data = self.datas.iloc[item]
        input_token = data['random_text'].strip()
        input_token = list(input_token)
        input_token = ['[CLS]'] + input_token + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_token)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        output_token = data['origin_text'].strip()
        output_token = list(output_token)
        output_token = ['[CLS]'] + output_token + ['[SEP]']
        output_ids = self.tokenizer.convert_tokens_to_ids(output_token)
        label = [int(x) for x in data['label'].strip().split()]
        label = [0] + label + [0]

        input_ids = torch.from_numpy(np.asarray(input_ids))
        input_mask = torch.from_numpy(np.asarray(input_mask))
        segment_ids = torch.from_numpy(np.asarray(segment_ids))
        output_ids = torch.from_numpy(np.asarray(output_ids))
        label = torch.from_numpy(np.asarray(label))
        return input_ids, input_mask, segment_ids, output_ids, label

def collate_fn(batch_data):
    batch_input_ids = [data[0] for data in batch_data]
    batch_input_mask = [data[1] for data in batch_data]
    batch_segment_ids = [data[2] for data in batch_data]
    batch_output_ids = [data[3] for data in batch_data]
    batch_labels = [data[4] for data in batch_data]

    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True)
    batch_input_mask = pad_sequence(batch_input_mask, batch_first=True)
    batch_segment_ids = pad_sequence(batch_segment_ids, batch_first=True)
    batch_output_ids = pad_sequence(batch_output_ids, batch_first=True)
    batch_labels = pad_sequence(batch_labels, batch_first=True)
    return batch_input_ids, batch_input_mask, \
                      batch_segment_ids, batch_output_ids, batch_labels
