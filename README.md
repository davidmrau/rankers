# Rankers Library

This library supports training, evaluation and encoding (indexing) with various neural state-of-the-art transformer-based IR models in python.


## Library Structure:
```
data_reader.py			- data_loader taking care of the pre-processing, read files iteratively (without loading them into memory)
datasets.json			- defines datasets and file paths 
decode.py			- decoding function
eval_model.py			- evaluation function
examples/			- example files
file_interface.py		- reads files into abstract class 
metrics.py 			- evaluation with trec_eval
models/				- base directory containing model files
	__init__.py		- scans model directory for model classes
	[model_name].py 	- defines model
run.py				- main file evokes training, evaluation and encoding
tools/				- various python tool scripts such as significance testing, etc.
train_model.py			- training function	
utils.py			- losses, RegScheduler, util functions
```

## Getting Started 

First, install the dependencies via pip install:

```python
pip3 install -r requirements.txt
```

## Flags 
We list all parameter flags with their description:


```
usage: run.py [-h] --model
options:
  -h, --help            show this help message and exit
  --model {Bert,BiEncoder,Bigbird,BowBert,Contriever,CrossEncoder,CrossEncoder2,DistilDot,DUOBert,Electra,IDCM,LongformerQA,Longformer,MiniLM12,MiniLM6,MonoLarge,nboostCrossEncoder,SentenceBert,ShuffleBert,SortBert,SparseBert,SpladeCocondenserEnsembleDistil,TinyBert}
                        Model name defined in model.py
  --exp_dir EXP_DIR     Base directory where files will be saved to.
  --dataset_test {example, 2019_pass,2019_doc,2020_pass,2020_doc,2021_pass,2021_doc,2022_doc,clueweb,robust,robust_100_callan,robust_100_kmeans}
                        Test dataset name defined in dataset.json
  --dataset_train {example, pass,doc,doc_tfidf}
                        Train dataset name defined in dataset.json
  --encode ENCODE       Path to file to encode. Format "qid did ".
  --add_to_dir ADD_TO_DIR
                        Will be appended to the default model directory
  --no_fp16             Disable half precision training.
  --mb_size_test MB_SIZE_TEST
                        Test batch size.
  --num_epochs NUM_EPOCHS
                        Number of training epochs.
  --max_inp_len MAX_INP_LEN
                        Max. total input length.
  --max_q_len MAX_Q_LEN
                        Max. Query length.
  --mb_size_train MB_SIZE_TRAIN
                        Train batch size.
  --single_gpu          Limit training to a single gpu.
  --eval_metric EVAL_METRIC
                        Evaluation Metric.
  --learning_rate LEARNING_RATE
                        Learning rate for training.
  --checkpoint CHECKPOINT
                        Folder of model checkpoint (will be loaded with huggingfaces .from_pretrained)
  --truncation_side {left,right}
                        Truncate from left or right
  --continue_line CONTINUE_LINE
                        Continue training in triples file from given line
  --save_last_hidden    Saves last hiden state under exp_dir/model_dir/last_hidden.p
  --aloss_scalar ALOSS_SCALAR
                        Loss scalar for the auxiliary sparsity loss.
  --aloss               Using auxilliary sparsity loss.
  --tf_embeds           [Experimental] Add term frequencies to input embeddings.
  --sparse_dim SPARSE_DIM
                        Dimensionality of the sparsity layer.
  --no_pos_emb          [Experimental] Removes the position embedding.
  --shuffle             [Experimental] Shuffles training and test tokens (after tokenization)
  --sort                [Experimental] Sortes document tokens in descending order by tokenid.
  --eval_strategy {first_p,last_p,max_p}
                        Evaluation strategy.
  --keep_q              [Experimental] Remove all but query terms in document.
  --drop_q              [Experimental] Removes all query terms from document.
  --preserve_q          [Experimental]
  --mse_loss            [Experimental]
  --rand_passage        [Experimental] Select a random passage of length "max_q_len" from entire input.
```



## Evaluation 
To evaluate a model run `run.py` providing the dataset that you want to be used with the flag `--dataset_test [dataset_name]`. 

Implemented datasets are 
`example`,
`2019_pass,2019_doc` ,
`2020_pass`,`2020_doc`,
`2021_pass`,
`2021_doc`,
`2022_doc`,
`clueweb`,
`robust`,
`robust_100_callan`,
`robust_100_kmeans`.

Please make sure all corresponding files (to be found in `datasets.json`) exist or to update the paths accordingly.

Example testing the `CrossEncoder` on the `example` dataset: 

```python
python3 run.py \
	--model 'CrossEncoder' \
	--dataset_test 'example' \
	--max_inp_len 512 \
	--mb_size_test 128 \
	--exp_dir '/tmp/example/'
```

### Adding a new Test Dataset:

New datasets with their respective file paths can be added in `datasets.json` and subsequently be loaded using `--dataset_test new_dataset`. The dataset name also needs to be added to the choices argument list in `run.py` to the flag `--datasets_test`.  In our example case this would be `new_dataset`. Test Datasets follow the format:

```
{
	{ 'test':
		'new_dataset': 
			'qrels': 'qrels_path',
			'trec_run': 'trec_run_path',
			'queries': 'queries_path',
			'docs': 'doc_path',
		
	},
	
	{
	...
	}
}
```

We provide examplary inputs of each file in the following:

`qrels`:

```
23849	0	1020327	2
23849	0	1034183	3
23849	0	1120730	0
23849	0	1139571	1
...
```


`trec_run` (contains ids of documents and queries that will be scored):

```
1030303 Q0 1038342
1030303 Q0 1154757
1030303 Q0 1161432
1030303 Q0 1161439
...
```

`queries`:

```
121352	define extreme
634306	what does chattel mean on credit history
920825	what was the great leap forward brainly
510633	tattoo fixers how much does it cost
...
```
`docs`:

```
0	The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.
1	The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.
2	Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.
...
```



## Training
To train a model run `run.py` providing the dataset you want to train on. This is done by adding the flag `--dataset_train [dataset_name]` 


 which invoces the training loop and will perform evaluation after each training epoch. The number of batch update steps that make up an epoch are defined in `train_model.py` alsongside other parameters such as `evaluate_every`, etc..

Note, the trainer will by default train with half precision (fp16). Use the flag `--no_fp16` to train with full floating point precision. 

This example trains a `CrossEncoder` defined in `models/cross_encoder.py` on the example training dataset (`example`) defined in `dataset.json` and evaluates the model in between the epochs on the `example` datasetset.

```python
python3 run.py \
	--dataset_train 'example' \
	--dataset_test 'example' \
	--model 'CrossEncoder' \
	--exp_dir '/tmp/example/' \
	--mb_size_train 128 \
	--mb_size_test 128 \
	--learning_rate 0.000003 \
	--max_inp_len 512

```

### Adding a new Training Dataset:

New datasets with their respective file paths can be added in `datasets.json` and subsequently be loaded using `--dataset_train new_dataset`. The dataset name also needs to be added to the choices argument list in `run.py` to the flag `--datasets_train`. In our example case this would be `new_dataset`.
Training Datsets follow the format:

```
{
	{ 'train':
		
		'new_dataset': 
			'triples': 'triples_path',
			'queries': 'queries_path',
			'docs': 'doc_path',
	},

	{
	...
	}
}
```

We provide examplary inputs of each file in the following:

`triples` (Query, Relevant Doc., Non-relevant Doc.):

```
662731	193249	2975302
527862	1505983	2975302
984152	2304924	3372067
...
```

`queries`:

```
121352	define extreme
634306	what does chattel mean on credit history
920825	what was the great leap forward brainly
...
```
`docs`:

```
0	The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.
1	The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science.
2	Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade.
...
```



## Encoding

Example encoding `examples/docs.tsv` using the  `CrossEncoder`. 

```python
python3 run.py \
	--model 'SparseBert' \
	--encode 'examples/docs.tsv' \
	--max_inp_len 512 \
	--mb_size_test 128 \
	--exp_dir '/tmp/example/'
```

This will save embeddings as numpy arrays in a dict under `/tmp/experiments_folder/example_docs.tsv.encoded.p` with the format:

```
{
	"doc_id_1": numpy.ndarray(embedding_doc_id_1),
	"doc_id_1": numpy.ndarray(embedding_doc_id_2),
	"doc_id_1": numpy.ndarray(embedding_doc_id_3),
	...
}
```

The document to be decoded should be in the following format: 

```
121352	define extreme
634306	what does chattel mean on credit history
920825	what was the great leap forward brainly
...
```

## Models 

Currently models `Bert`, `BiEncoder`, `Bigbird`, `BowBert`, `Contriever`, `CrossEncoder`, `CrossEncoder2`, `DistilDot`, `DUOBert`, `Electra`, `IDCM`, `LongformerQA`, `Longformer`, `MiniLM12`, `MiniLM6`, `MonoLarge`, `nboostCrossEncoder`, `SentenceBert`, `ShuffleBert`, `SortBert`, `SparseBert`, `SpladeCocondenserEnsembleDistil`, and `TinyBert` are implemented. 

All models are defined in the folder `models/`. The file `models/__init__.py` will automatically read all  python files (`*.py`) and import all (model) classes defined in `models/[model_name.py]`. The models can then be used by running:

`python3 run.py --model [ModelClassName]`

Our library uses a python wrapper around each model which follows the following Template:

```python
class ModelClassName():

    def __init__(self, kwargs):
        self.kwargs = kwargs
        
        # instantiate your tokenizer here
        self.tokenizer 
        # instantiate your model here
        self.model  
        # select type of model either 'cross'  or 'bi'
        self.type  

    def get_scores(self, features, index):
        encoded_input = features['encoded_input'][index]
        
        # your code for calling the models forward-pass
        scores =   # model inference. for example self.model(**encoded_input.to('cuda'))
        
        return_dict = {}
        return_dict['scores'] = scores
        return return_dict
```

- `model_type`: `bi` or `cross` (Whether the model is a Cross-Encoder or Bi-Encoder which defines preprocessing and scoring of documents and queries.)


- `tokenizer`: Tokenizer

- `model`: Model

- `get_scores`: Wrapper around the models forward and returns bare scores for Cross-Encoders or an embedding for Bi-Encoders

The model class name also needs to be added to the choices argument list in `run.py` to the flag `--model` in our example case this would be `ModelClassName`.


## Loading model checkpoints

Model checkpoints can be conveniently loaded by adding the flag `--checkpoint [MODEL_DIR]` to the model directory. Models will be loaded using huggingface's .from_pretrained method. Therefore, only models that inherit from `PreTrainedModel` can be loaded. An example of how to build a custom model that is compatible can be found under `examples/custom_model.py`.

An example of how to load a checkpoint:

```python
python3 run.py \
	--model 'CrossEncoder' \
	--dataset_test 'example' \
	--max_inp_len 512 \
	--mb_size_test 128 \
	--exp_dir '/tmp/example/' \
	--checkpoint '/folder/to/model/'
```





