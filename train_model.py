
from performance_monitor import PerformanceMonitor
import torch
from tqdm import tqdm
from util import print_message, PostingBalance, FLOPS
from eval_model import eval_model
import amp
import time
from util import DistilMarginMSE

def train_model(ranker, dataloader_train, dataloader_test, qrels_file, criterion, optimizer, scheduler, reg_d, reg_q, model_dir, training_steps=150000, log_every=500, save_every=10000, aloss=False, fp16=True, wandb=None, eval_every=2500, accumulation_steps=1):
    mpm = amp.MixedPrecisionManager(fp16)
    batch_iterator = iter(dataloader_train)
    ranker.model.train()
    for training_step in tqdm(range(1, training_steps+1), desc='Training Loop'):
        # get train data
        try:
            features = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader_train)
            features = next(batch_iterator)
        with mpm.context():
            out_1, out_2 = ranker.get_scores(features, index=0), ranker.get_scores(features, index=1)
            scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
            if isinstance(criterion, DistilMarginMSE):
                target_pos, target_neg = features['teacher_pos_scores'].to('cuda'), features['teacher_neg_scores'].to('cuda')

            #    train_loss = criterion(out_1scores_doc_1, scores_doc_2, torch.tensor(features['labels_1'], device='cuda'), torch.tensor(features['labels_2'], device='cuda'))

                train_loss = criterion(scores_doc_1, scores_doc_2, target_pos, target_neg)
            elif 'bi' in ranker.type:
                train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
                #train_loss = logsoftmax(torch.cat([scores_doc_1.unsqueeze(1), scores_doc_2.unsqueeze(1)], dim=1))
                #train_loss = torch.mean(-train_loss[0,:])
                
            elif 'cross' in ranker.type:
                scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                train_loss = criterion(scores, features['labels'].long().to('cuda'))
            else:
                raise NotImplementedError()
            if aloss:

                aloss_scalar_d = reg_d.step()
                aloss_scalar_q = reg_q.step()
                #l1_loss = out_1['reg_queries'] + ( (out_1['reg_docs'] + out_2['reg_docs']) /2)
                l1_loss = aloss_scalar_d * ( (out_1['reg_docs'] + out_2['reg_docs']) /2)
                l1_loss += aloss_scalar_q * out_1['reg_queries'] 

                train_loss += l1_loss
            if torch.isnan(train_loss).sum()> 0:
                raise ValueError('loss contains nans, aborting training')
                exit()
            if accumulation_steps > 1:
               train_loss /= accumulation_steps
            mpm.backward(train_loss)
            
            if training_step % accumulation_steps == 0:
                mpm.step(optimizer)
                scheduler.step()
            if training_step % log_every == 0:
                    del out_1['scores']
                    print_dict = {"train_loss": train_loss}
                    print_dict.update(out_1)
                    if wandb:
                        wandb.log(print_dict)
                    print_message(print_dict)

            if training_step % eval_every == 0:
                eval_model(ranker, dataloader_test, qrels_file, model_dir, suffix=training_steps, wandb=wandb)

            if training_step % save_every == 0:
                if hasattr(ranker.model, 'module'):
                    ranker.model.module.save_pretrained(f'{model_dir}/model_{training_step}')
                else:
                    ranker.model.save_pretrained(f'{model_dir}/model_{training_step}')
                print('saving_model')
