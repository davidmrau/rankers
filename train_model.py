
from performance_monitor import PerformanceMonitor
import torch
from tqdm import tqdm
from util import print_message, PostingBalance, FLOPS
from eval_model import eval_model


def train_model(ranker, dataloader_train, dataloader_test, qrels_file, criterion, optimizer, scaler, scheduler, reg,  model_dir, num_epochs=40, epoch_size=1000, log_every=10, save_every=1, aloss=False, aloss_scalar=None, fp16=True):
    batch_iterator = iter(dataloader_train)
    total_examples_seen = 0
    ranker.model.train()
    flops = PostingBalance()
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
                scores_doc_1, scores_doc_2 = out_1['scores'], out_2['scores']
                optimizer.zero_grad()
                #if args.mse_loss:
                #    train_loss = criterion(scores_doc_1, scores_doc_2, torch.tensor(features['labels_1'], device='cuda'), torch.tensor(features['labels_2'], device='cuda'))
                if ranker.type == 'bi':
                    train_loss = criterion(scores_doc_1, scores_doc_2, features['labels'].to('cuda'))
                    #train_loss = logsoftmax(torch.cat([scores_doc_1.unsqueeze(1), scores_doc_2.unsqueeze(1)], dim=1))
                    #train_loss = torch.mean(-train_loss[0,:])
                    
                elif ranker.type == 'cross':
                    scores = torch.stack((scores_doc_2, scores_doc_1),1 )
                    train_loss = criterion(scores, features['labels'].long().to('cuda'))
                else:
                    raise NotImplementedError()
                if aloss:

                    #l1_loss = (out_1['l1_queries']).mean() + (((out_1['l1_docs'])).mean() +  (out_2['l1_docs']).mean() / 2)
                    l1_loss = flops(out_1['l1_queries']) + (  ( flops(out_1['l1_docs']) + flops(out_2['l1_docs']) ) / 2)
                    l1_loss = l1_loss * aloss_scalar
                    l0_loss = (( out_1['l0_docs'] + out_2['l0_docs']) /2 )
                    used_dims = (( out_1['used_dims'] + out_2['used_dims']) /2 )
                    aloss_scalar = reg.step()
                else:
                    l1_loss = 0
                    l0_loss = 0
                    used_dims = 0
                if torch.isnan(train_loss).sum()> 0:
                    raise ValueError('loss contains nans, aborting training')
                    exit()
                train_loss += l1_loss
                total_examples_seen += scores_doc_1.shape[0]
                # scaler.scale(loss) returns scaled losses, before the backward() is called
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += train_loss.item()
                if mb_idx % log_every == 0:
                        print(f'MB {mb_idx + 1}/{epoch_size}')
                        print_message('examples:{}, train_loss:{:.6f}, l1_loss:{:.6f}, l0_loss:{:.2f}, used:{:.2f}'.format(total_examples_seen, train_loss, l1_loss, l0_loss, used_dims))
                mb_idx += 1

        print_message('epoch:{}, av loss:{}'.format(ep_idx + 1, epoch_loss / (epoch_size) ))

        eval_model(ranker, dataloader_test, qrels_file, model_dir, suffix=ep_idx+1)

        print('saving_model')

        if ep_idx % save_every == 0:
            if hasattr(ranker.model, 'module'):
                ranker.model.module.save_pretrained(f'{model_dir}/model_{ep_idx+1}')
            else:
                ranker.model.save_pretrained(f'{model_dir}/model_{ep_idx+1}')
