import torch
import torch.nn as nn
import wandb

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import argparse
import numpy as np
import os
from nltk.corpus import stopwords
from transformers.optimization import get_linear_schedule_with_warmup
from models.cnnmodel import CNNModel, CNNConfig

class ImportanceDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, max_length, stopwords_to_zero=True):
        super().__init__()
        self.filename = filename
        self.max_length = max_length
        self.stopwords_to_zero = stopwords_to_zero
        self.stop_words = set(stopwords.words('english'))

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Extract input and target from the JSON object

                term_importance_tuples = data['plm_bert_tokenized']
                tokens, importance = list(), list()
                for term, score in term_importance_tuples:
                    if self.stopwords_to_zero:
                        if term in self.stop_words:
                            score = 0
                    tokens.append(term)
                    importance.append(score)
                tokens, importance =  tokens[:self.max_length], importance[:self.max_length] 
                normed_importance = self.norm(importance)
                if normed_importance == None :
                    continue
                yield tokens, normed_importance


    def norm(self, x):
        x = np.array(x)
        x_min = min(x)
        x_max = max(x)
        den = (x_max - x_min)

        if den == 0:
            return None
        else:
            x_normalized = (x - x_min) / den
            return x_normalized.tolist()



def pad_and_convert_to_tensor(data, padding_value=0, max_length=None):
    # Determine the maximum length of the sequences in the data
    if max_length is None:
        max_length = max(len(seq) for seq in data)
    
    # Pad each sequence to the maximum length
    padded_data = [list(seq[:max_length]) + [padding_value] * (max_length - len(seq)) for seq in data]

    # Convert the padded data to a PyTorch tensor
    tensor_data = torch.tensor(padded_data)
    return tensor_data

def collate_fn(batch):
    input_tokens, importance = zip(*batch)
    input_token_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in input_tokens]
    # Pad input token IDs to same length using tokenizer.pad
    padded_input_token_ids = pad_and_convert_to_tensor(input_token_ids)  # pad importance to fixed length
    # Convert to PyTorch tensor
    #importance_ext = pad_and_convert_to_tensor(importance, max_length=padded_input_token_ids.shape[1])  # pad importance to fixed length
    importance_ext = pad_and_convert_to_tensor(importance)  # pad importance to fixed length
    #return padded_input_token_ids, norm(importance_ext)
    return padded_input_token_ids, importance_ext


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    
    # Hyperparameters
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of training steps (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training (default: 1024)')
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[1, 15, 63],
                        help='Filter sizes for convolutional layers (default: [1, 15, 63])')
    parser.add_argument('--num_filters', type=int, default=4,
                        help='Number of filters for each filter size (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--max_length', type=int, default=25000,
                        help='Maximum input sequence length (default: 25000)')
    parser.add_argument('--train_jsonl', type=str, required=True,
                        help='Training data in jsonl format')
    parser.add_argument('--val_jsonl', type=str, required=True,
                        help='Validation data in jsonl format')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Train the model
    num_steps = args.num_steps
    batch_size = args.batch_size
    filter_sizes  = args.filter_sizes
    num_filters = args.num_filters
    lr = args.lr
    max_length = args.max_length
    # Load data
    train_dataset = ImportanceDataset(args.train_jsonl, max_length=max_length)
    valid_dataset = ImportanceDataset(args.val_jsonl, max_length=max_length)
    
    patience = 3
    eval_every = 50

    str_fs = "_".join([str(x) for x in filter_sizes])
    model_folder = f'num_steps_{num_steps}_bs_{batch_size}_fs_{str_fs}_filters_{num_filters}_lr_{lr}_pat_{patience}_max_len_{max_length}'
    path = f"/scratch/drau/models/extractor_passage/{model_folder}/"

    # create folder
    os.makedirs(path, exist_ok=True)

    wandb.login()
    wandb.init( project="pretrain_term_exractor_passage", config=args)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('huggingface/bert-base-uncased/')
    # Initialize the CNN model
    cfg = CNNModelConfig(filter_sizes=args.filter_sizes, num_filters=args.num_filters)
    model = CNNModel(cfg)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    #criterion = nn.KLDivLoss(reduction = 'batchmean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=250, num_training_steps=150000)
    scaler = torch.cuda.amp.GradScaler(enabled=True)


    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        device = torch.device('cuda')
    else:
        print("CUDA is not available! Using CPU.")
        device = torch.device('cpu')
    # Create a DataParallel model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.to(device))

    # Initialize variables for early stopping
    best_loss = np.inf
    early_stop_counter = 0

    # Training loop
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)
    val_loader = DataLoader(valid_dataset, batch_size*2, collate_fn=collate_fn, num_workers=1)
    for steps, (tokens, importance) in enumerate(train_loader):
        # Clear gradients
        optimizer.zero_grad()
        # Get BERT embeddings for the tokens
        #embeddings = word_embeddings(tokens.to(device))
        # Pass the embeddings through the CNN model
        scores, _ = model(tokens.to(device))
        mask = tokens != 0
        loss = criterion(scores, importance.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if torch.isnan(loss).sum()> 0:
            raise ValueError('loss contains nans, aborting training')
            print(tokens)
            exit()
        train_loss = loss.item()
        wandb.log({"train_loss": train_loss})

        val_loss = 0
        val_steps = 0

        if steps % eval_every == 0 and steps != 0: 
            # Validation loop
            model.eval()
            with torch.no_grad():
                for tokens, importance in val_loader:
                    #embeddings = word_embeddings(tokens.to(device))
                    scores, _ = model(tokens.to(device))
         # Calculate loss and update parameters
                    for j, (i, p, t) in enumerate(zip(tokens[0], scores[0], importance[0])):
                        token = [tokenizer.decode([i])]
                        print(f'token: {token}\tpredicted: {p:.3f}\ttarget: {t:.3f}\n')
                        #if j > 150:
                        #    break
                    mask = tokens != 0
                    loss = criterion(scores, mask.to(device))

                    val_loss += loss.item()
                    val_steps += 1
            wandb.log({"val_loss": val_loss/val_steps})

            # Check if the current validation loss is the best seen so far
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                if hasattr(model, 'module'):
                    model.module.save_pretrained(path)
                else:
                    model.save_pretrained(path)
            else:
                early_stop_counter += 1
            
            # Check if early stopping criteria are met
            if early_stop_counter >= patience:
                print(f"Early stopping after {steps} steps.")
                exit()






