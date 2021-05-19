

class Config(object):

    def __init__(self):
        self.bert_path = '../pretrain_model/chinese_wwm_ext_pytorch'
        self.vocab_file = '../pretrain_model/chinese_wwm_ext_pytorch/vocab.txt'
        self.hidden_size = 50
        self.embedding_size = 768
        self.device = 'cuda'
        self.lr = 0.0001
        self.epoch = 5
        self.batch_size = 12
