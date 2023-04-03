import torch
import pickle
from tqdm import tqdm


def encode(ranker, encode_file, dataloader, model_dir, eval_strategy='first_p'):
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
                 

        

    #if encode_query: 
    #    f = open(f"{model_dir}_query_encoded.tsv", 'w', encoding='utf-8')
    #else:
    #    f = gzip.open(f"{model_dir}_docs_encoded.tsv.gz", 'wt', encoding='utf-8')
#            # splade decode docs
#
#           weight_range = 5
#            quant_range = 256
#            # decode and print random sample
#            if num_i == 0:
#                idxs = random.sample(range(len(ids)), 1)
#                for idx in idxs:
#                    print(ids[idx], tokenizer.decode(features[1]['input_ids'][idx]))
#
#            for id_, latent_term in zip(ids, latent_terms):
#                if encode_query:
#                    pseudo_str = []
#                    for tok, weight in latent_term.items():
#                        #weight_quanted = int(np.round(weight/weight_range*quant_range))
#                        weight_quanted = int(np.round(weight*100))
#                        pseudo_str += [tok] * weight_quanted
#                    latent_term = " ".join(pseudo_str)
#                    f.write(f"{id_}\t{latent_term}\n")
#                else:
#                    f.write( json.dumps({"id": id_, "vector": latent_term }) + '\n')




