import torch.nn as nn
import torch
import datetime


class MarginMSELoss(nn.Module):
    def __init__(self):
        super(MarginMSELoss, self).__init__()

    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        A Margin-MSE loss, receiving 2 scores and 2 labels and it computes the MSE of the respective margins.
        All inputs should be tensors of equal size
        """
        loss = torch.mean(torch.pow((scores_pos - scores_neg) - (label_pos - label_neg),2))
        return loss

class FLOPS:
    def __call__(self, x):
        return torch.sum(torch.mean(torch.abs(x), dim=0) ** 2)

class PostingBalance:

    def cv_squared(self, x):
        eps = 1e-10
        return x.float().var() / (x.float().mean()**2 + eps)

    def __call__(self, x):
        return self.cv_squared(x.sum(0))


class DistilMarginMSE(nn.Module):
    """MSE margin distillation loss from: Improving Efficient Neural Ranking Models with Cross-Architecture
    Knowledge Distillation
    link: https://arxiv.org/abs/2010.02666
    """

    def __init__(self):
        super(DistilMarginMSE, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, pos_scores, neg_scores, teacher_pos_scores, teacher_neg_scores):
        margin = pos_scores - neg_scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores
        return self.loss(margin.squeeze(), teacher_margin.squeeze()).mean()  # forces the margins to be similar

def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class RegWeightScheduler:
    """same scheduling as in: Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T
        self.t = 0
        self.lambda_t = 0

    def step(self):
        """quadratic increase until time T
        """
        if self.t >= self.T:
            pass
        else:
            self.t += 1
            self.lambda_t = self.lambda_ * (self.t / self.T) ** 2
        return self.lambda_t

    def get_lambda(self):
        return self.lambda_t




#from fairseq.models.roberta import RobertaModel
#    elif 'roberta.shuffle' in model_name or 'roberta.base.orig' == model_name:
#
#        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#        model = AutoModelForSequenceClassification.from_pretrained(f'models/{model_name}')
#        encoding = 'cross'
#
#        def get_scores(model, features, index):
#            encoded_input = features['encoded_input'][index]
#            out_raw = model(**encoded_input.to('cuda'))
#            scores = out_raw.logits[:, 1]
#            return_dict = {}
#            return_dict['scores'] = scores
#            return return_dict
#    elif 'roberta.base.nopos' in model_name :
#        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#        model = AutoModelForSequenceClassification.from_pretrained(f'models/{model_name}')
#        encoding = 'cross'
#
#        def get_scores(model, features, index):
#            encoded_input = features['encoded_input'][index]
#            out_raw = model(**encoded_input.to('cuda'))
#            scores = out_raw.logits[:, 1]
#            return_dict = {}
#            return_dict['scores'] = scores
#            return return_dict
#        
#
#
#    elif 'bert_tf' == model_name:
#        model_name = "bert-base-uncased"
#        tokenizer = AutoTokenizer.from_pretrained(model_name)
#        model = AutoModelForSequenceClassification.from_pretrained(model_name)
#        #model = BERT_TF(BERT_TF_Config())
#        #model = BERT_TF()
#        encoding = 'cross'
#
#        def get_scores(model, features, index):
#            encoded_input = features['encoded_input'][index]
#            out_raw = model(**encoded_input.to('cuda'), position_embeds=features['tf_embeds'][index].to('cuda'))
#            scores = out_raw.logits[:, 1]
#            return_dict = {}
#            return_dict['scores'] = scores
#            return return_dict
#
#
#    elif 'tctcolbert' == model_name:
#        model_name = "castorini/tct_colbert-msmarco"
#        model = AutoModel.from_pretrained(model_name)
#        tokenizer = AutoTokenizer.from_pretrained(model_name)
#        encoding = 'bi'
#        prepend_type = True
#
#        def get_scores(model, features, index):
#            encoded_queries = features['encoded_queries']
#            encoded_docs = features['encoded_docs'][index]
#            emb_queries = model(**encoded_queries.to('cuda'))
#            emb_docs = model(**encoded_docs.to('cuda'))
#            emb_queries_mean = mean_pooling(emb_queries["last_hidden_state"], encoded_queries['attention_mask'])
#            emb_docs_mean = mean_pooling(emb_docs["last_hidden_state"][:, 4:, :], encoded_docs['attention_mask'][:, 4:])
#            scores = torch.bmm(emb_queries_mean.unsqueeze(1), emb_docs_mean.unsqueeze(-1)).squeeze()
#            return_dict = {}
#            return_dict['scores'] = scores
#            return return_dict





