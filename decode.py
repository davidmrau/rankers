import torch
import pickle
from tqdm import tqdm
import random

def decode(ranker, encode_file, dataloader, model_dir, weight_range=5, quant_range=256):
    encode_file_no_path = encode_file.split('/')[-1]
    out_file = open(f"{model_dir}/{encode_file_no_path}.decoded.tsv", 'w')
    ranker.model.eval() 
    emb_dict = {} 
    with torch.no_grad():
        for num_i, features in tqdm(enumerate(dataloader)):
            with torch.inference_mode():
                ids, encoded_input = features
                # decode and print random sample
                if num_i == 0:
                    idxs = random.sample(range(len(ids)), 1)
                    for idx in idxs:
                        print(ids[idx], ranker.tokenizer.decode(features[1]['input_ids'][idx]))
                embs = ranker.decode(encoded_input)

                for id_, latent_term in zip(ids, embs):
                    pseudo_str = []
                    for tok, weight in latent_term.items():
                        weight_quanted = round( weight / weight_range*quant_range )
                        #weight_quanted = int(np.round(weight*100))
                        pseudo_str += [tok] * weight_quanted
                    latent_term = " ".join(pseudo_str)
                    out_file.write(f"{id_}\t{latent_term}\n")
                    #f.write( json.dumps({"id": id_, "vector": latent_term }) + '\n')

