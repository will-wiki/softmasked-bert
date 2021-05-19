
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model.BiGRU_Detector import BiGRU_Detector

class SoftMasked_Bert(nn.Module):

    def __init__(self, config):
        super(SoftMasked_Bert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path).to(self.config.device)
        self.embedding = self.bert.embeddings
        self.bert_encoder = self.bert.encoder
        self.input_size = self.config.embedding_size
        self.tokenizer = BertTokenizer.from_pretrained(self.config.vocab_file)
        self.masked_e = self.embedding(torch.tensor([[self.tokenizer.mask_token_id]], dtype=torch.long).to(self.config.device))
        self.detector = BiGRU_Detector(self.input_size, self.config.hidden_size)
        self.linear = nn.Linear(self.config.embedding_size, self.tokenizer.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        print()

    def forward(self, input_ids, input_mask, segment_ids):
        bert_embedding = self.embedding(input_ids)
        p = self.detector(bert_embedding)
        soft_bert_embedding = p * self.masked_e + (1 - p) * bert_embedding
        bert_out = self.bert_encoder(soft_bert_embedding)
        h = bert_out[0] + bert_embedding
        out = self.softmax(self.linear(h))
        return out, p

if __name__ == "__main__":
    from config import Config
    config = Config()
    model = SoftMasked_Bert(config)
    print()
