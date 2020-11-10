from transformers import BertConfig, BertModel, BertTokenizer

config = BertConfig.from_json_file("../models/bert/hottoSNS-bert-pytorch/config.json")
model = BertModel.from_pretrained("../models/bert/hottoSNS-bert-pytorch/hottoSNS-bert-pytorch.bin", config=config)
bert_tokenizer = BertTokenizer("../models/bert/hottoSNS-bert-pytorch/vocab.txt",do_lower_case=False, do_basic_tokenize=False)

print(model)