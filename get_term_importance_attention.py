from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch

from torch.utils.data import DataLoader


in_f = 'data/msmarco/collection.tsv'
out_f = open(f'{in_f}.tia_bert', 'w', encoding='utf-8')

t = AutoTokenizer.from_pretrained('bert-base-uncased')

class TSV(torch.utils.data.IterableDataset):
	def __init__(self, f):
		super(TSV).__init__()
		self.f = open(f, encoding='utf-8')

	def __iter__(self):
		for line in self.f:
			id_, text = line.strip().split('\t')
			yield text
			
	def collate_fn(self, data):
		return t(data, padding=True, truncation=True, return_tensors='pt')

dataset = TSV(in_f)
dataloader = DataLoader(dataset, batch_size=512, collate_fn=dataset.collate_fn, num_workers=1)

config = AutoConfig.from_pretrained('bert-base-uncased')
config.num_hidden_layers = 1

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
model = model.to('cuda')
model = torch.nn.DataParallel(model)

for data in dataloader:
    data = data.to('cuda')
    input_ids, attention_mask, token_type_ids = data['input_ids'], data['attention_mask'], data['token_type_ids']
    attentions = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,  output_attentions=True).attentions[0]
    mean_attentions = (attentions.mean(1) * attention_mask.unsqueeze(1)).sum(1)
    vals, idxs = mean_attentions[:,1:].topk(input_ids.shape[1]-1)
    for i, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist())):
        out_f.write(t.decode([input_ids[i, t_idx+1] for t_idx in idx], skip_special_tokens=True) + '\n')
