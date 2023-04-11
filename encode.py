import torch
import pickle
from tqdm import tqdm


def encode(ranker, encode_file, dataloader, model_dir):
    encode_file_no_path = encode_file.split('/')[-1]
    emb_file = f"{model_dir}/{encode_file_no_path}.encoded.p"
    ranker.model.eval() 
    emb_dict = {} 
    with torch.no_grad():
        for num_i, features in tqdm(enumerate(dataloader)):

            with torch.inference_mode():
                ids, encoded_input = features
                embs = ranker.encode(encoded_input)
                for id_, emb_ in zip(ids, embs.detach().cpu().numpy()):
                    emb_dict[id_] = emb_

        pickle.dump(emb_dict, open(emb_file, 'wb'))
                 
