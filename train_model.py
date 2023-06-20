
from performance_monitor import PerformanceMonitor
import torch
from tqdm import tqdm
from util import print_message, PostingBalance, FLOPS
from eval_model import eval_model

from util import DistilMarginMSE

def train_model(ranker, dataloader_train, dataloader_test, qrels_file, criterion, optimizer, scaler, scheduler, reg_d, reg_q, model_dir, num_epochs=40, epoch_size=100, log_every=100, save_every=1, aloss=False, fp16=True, wandb=None, eval_every=1):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    ranker.model.train()
    for ep_idx in range(num_epochs):
        print('epoch', ep_idx)
        # TRAINING
        epoch_loss = 0.0
        mb_idx = 0
        while mb_idx  <   epoch_size:
            # get train data
            try:
                features = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader_train)
                continue
            with torch.cuda.amp.autocast(enabled=fp16):
                out_1, out_2 = ranker.get_scores(features, index=0), ranker.get_scores(features, index=1)
                optimizer.zero_grad()
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
                total_examples_seen += scores_doc_1.shape[0]
                # scaler.scale(loss) returns scaled losses, before the backward() is called
                if torch.isnan(train_loss).sum()> 0:
                    raise ValueError('loss contains nans, aborting training')
                    exit()
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += train_loss.item()
                if mb_idx % log_every == 0:
                        print(f'MB {mb_idx + 1}/{epoch_size}')
                        del out_1['scores']
                        print_dict = {"train_loss": train_loss}
                        print_dict.update(out_1)
                        if wandb:
                            wandb.log(print_dict)
                        print_message(print_dict)
                mb_idx += 1

        print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, epoch_loss / (epoch_size) ))
        if ep_idx % eval_every == 0:
            eval_model(ranker, dataloader_test, qrels_file, model_dir, suffix=ep_idx+1, wandb=wandb)

        print('saving_model')

        if ep_idx % save_every == 0:
            if hasattr(ranker.model, 'module'):
                ranker.model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')
            else:
                ranker.model.save_pretrained(f'{model_dir}/model_{ep_idx+1}')
