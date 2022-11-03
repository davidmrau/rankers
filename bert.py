import torch

from torch import nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
#from sparse import *

class BERT(nn.Module):

	def __init__(self, dropout_p, freeze_bert=False, aggregation='cls', train_idf=False, scalar_init=1, idf=None, num_labels=1, mlm=False, no_pos_emb=False, all_attention_uniform=False, first_attention_uniform=False ):
		super(BERT, self).__init__()
		#self.bert = BertModel.from_pretrained('/project/draugpu/pretrain_bert_ranking/vanilla_12/checkpoint-1000/')

                #cfg = BertConfig(all_attention_uniform=all_attention_uniform, first_attention_uniform=first_attention_uniform)
                #print(cfg)
                #exit()
		#self.bert = BertModel.from_pretrained('bert-base-uncased', config=config)
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		#self.bert = BertModel(BertConfig())
		#print('!!!! from Scratch !!!')	
		#self.bert = BertModel()
		#old_ttids = self.bert.embeddings.token_type_embeddings.weight.data
		#self.bert.embeddings.token_type_embeddings = nn.Embedding(3, 768)
		#self.bert.embeddings.token_type_embeddings.weight.data[:2] = old_ttids[:2]
		#self.bert = BertModel(BertConfig(type_vocab_size=3))
		self.rank = nn.Linear(768, num_labels)
		self.matching_mask = nn.Linear(768, 3)
		self.drop = nn.Dropout(p=dropout_p)
		self.aggregation = aggregation
		if mlm:
			self.mlm_head = BertOnlyMLMHead(BertConfig()) 
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False
		else:
			for param in self.bert.parameters():
				param.requires_grad = True
		if no_pos_emb:
			self.bert.embeddings.position_embeddings.weight.data = torch.zeros_like(self.bert.embeddings.position_embeddings.weight.data)
			#print('!!! training pos embedding!!!!!')
			self.bert.embeddings.position_embeddings.weight.requires_grad = False
			print('!!Removing positional Embeddings!!!')


		if self.aggregation == 'idf':
			self.idf = nn.Embedding.from_pretrained(torch.FloatTensor(idf).unsqueeze(1), freeze=not train_idf)
			self.scalar = torch.nn.Parameter(torch.FloatTensor([scalar_init]), requires_grad=True)
			self.softmax = nn.Softmax(1)



		
	def forward(self, input_ids, attention_mask, token_type_ids, mask=None, labels=None, mlm=False, matching_mask=False, hidden_states=False, attentions=False):
		if hidden_states:
			bert_output = self.bert( input_ids, attention_mask, token_type_ids, output_hidden_states=True)
			return bert_output.hidden_states
		if attentions:
			bert_output = self.bert( input_ids, attention_mask, token_type_ids, output_attentions=True)
			return bert_output.attentions[0]

		# ignore cls
		if self.aggregation == 'sep' or self.aggregation == 'idf' or self.aggregation == 'mean' or self.aggregation == 'special':
			input_ids, attention_mask, token_type_ids = input_ids[:,1:], attention_mask[:, 1:], token_type_ids[:, 1:]

		bert_output = self.bert( input_ids, attention_mask, token_type_ids )
		last_hidden_state = bert_output.last_hidden_state #b x len x hidden	
		last_hidden_state = self.drop(last_hidden_state)
		if mlm:
        		sequence_output = bert_output[0]
        		return self.mlm_head(sequence_output)
		elif matching_mask:
			out = self.matching_mask((last_hidden_state * attention_mask.unsqueeze(2)).mean(1))
		elif self.aggregation == 'cls':
			out = last_hidden_state[:,0]

		elif self.aggregation == 'sep':
			lengths = (input_ids == 102).int().nonzero()[::2][:,1]

			mask = torch.zeros_like(input_ids, dtype=torch.bool)
			range_tensor = torch.arange(input_ids.shape[1]).unsqueeze(0)

			if input_ids.is_cuda:
				range_tensor = range_tensor.cuda()

			range_tensor = range_tensor.expand(lengths.size(0), range_tensor.size(1))
			mask = (range_tensor <  lengths.unsqueeze(1))
			out = (mask.unsqueeze(2) * last_hidden_state ).sum(dim = 1)

		elif self.aggregation == 'mean':
			out = self.rank((last_hidden_state * attention_mask.unsqueeze(2)).mean(1))


		elif self.aggregation == 'idf':
			idf_values = self.idf(input_ids) * attention_mask.unsqueeze(2)
			idf_values[torch.isnan(idf_values)] = 0
			idf_norm = self.softmax(self.scalar * idf_values)
			out = (last_hidden_state * idf_norm).sum(1)

		elif self.aggregation == 'special':
			mask = (input_ids == 30523) | (input_ids == 30522)
			out = (mask.unsqueeze(2) * last_hidden_state ).sum(dim = 1)
		elif self.aggregation == 'mm':
			mask = (input_ids == 30522)
			out = (mask.unsqueeze(2) * last_hidden_state ).sum(dim = 1)
		elif self.aggregation == 'm':
			mask = (input_ids == 30523)
			out = (mask.unsqueeze(2) * last_hidden_state ).sum(dim = 1)
		elif self.aggregation == 'matching_mask':	
			#out =
			pass
		else:
			raise NotImplementedError() 
		return self.rank(out)
		
