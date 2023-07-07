
from performance_monitor import PerformanceMonitor
import torch
from tqdm import tqdm
from util import print_message, PostingBalance, FLOPS
from eval_model import eval_model
import amp
import time
from util import DistilMarginMSE

def train_model(ranker, dataloader_train, dataloader_test, qrels_file, criterion, optimizer, scheduler, reg_d, reg_q, model_dir, training_steps=150000, save_every=10000, aloss=False, fp16=True, wandb=None, eval_every=2500, accumulation_steps=1):
    if eval_every >= 4:
        log_every = eval_every // 4
    else:
        log_every = eval_every
    save_every = eval_every
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
            torch.cuda.empty_cache()
            out_1, out_2 = ranker.get_scores(features, index=0), ranker.get_scores(features, index=1)
            if 'causallm' in ranker.type:
                train_loss = (out_1['loss'] + out_2['loss']) / 2
            elif isinstance(criterion, DistilMarginMSE):
                scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
                target_pos, target_neg = features['teacher_pos_scores'].to('cuda'), features['teacher_neg_scores'].to('cuda')
                train_loss = criterion(scores_doc_1, scores_doc_2, target_pos, target_neg)
            elif 'bi' in ranker.type:
                scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
                train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
                
            elif 'cross' in ranker.type:
                scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
                scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                train_loss = criterion(scores, features['labels'].long().to('cuda'))
            else:
                raise NotImplementedError()
            if aloss:

                aloss_scalar_d = reg_d.step()
                aloss_scalar_q = reg_q.step()
                l1_loss = aloss_scalar_d * ( (out_1['reg_docs'] + out_2['reg_docs']) /2)
                l1_loss += aloss_scalar_q * out_1['reg_queries'] 

                train_loss += l1_loss
            if torch.isnan(train_loss).sum()> 0:
                raise ValueError('loss contains nans, aborting training')
                exit()
            if accumulation_steps > 1:
               train_loss /= accumulation_steps
            mpm.backward(train_loss)
            torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.0) 
            if training_step % accumulation_steps == 0:
                mpm.step(optimizer)
                scheduler.step()
            if training_step % log_every == 0:
                    if 'scores' in out_1:
                        del out_1['scores']
                    print_dict = {"train_loss": train_loss}
                    print_dict.update(out_1)
                    print_dict.update({'batch': training_step})
                    if ranker.type == 'causallm':
                        print_dict.update({'ppl': torch.exp(train_loss)})
                    if wandb:
                        wandb.log(print_dict)
                    print_message(print_dict)
            del out_1
            del out_2
            del train_loss
            del features 
            torch.cuda.empty_cache()
            if training_step % eval_every == 0:
                eval_model(ranker, dataloader_test, qrels_file, model_dir, suffix=training_step, wandb=wandb)

            if training_step % save_every == 0:
                if hasattr(ranker.model, 'module'):
                    ranker.model.module.save_pretrained(f'{model_dir}/model_{training_step}')
                else:
                    ranker.model.save_pretrained(f'{model_dir}/model_{training_step}')
                print('saving_model')
