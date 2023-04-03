

import os, sys

dir_path = os.path.dirname(os.path.abspath(__file__))
files_in_dir = [f[:-3] for f in os.listdir(dir_path)
                if f.endswith('.py') and f != '__init__.py']
for f in files_in_dir:
    mod = __import__('.'.join([__name__, f]), fromlist=[f])
    to_import = [getattr(mod, x) for x in dir(mod)]
               # if isinstance(getattr(mod, x), type)]  # if you need classes only
    for i in to_import:
        try:
            setattr(sys.modules[__name__], i.__name__, i)
        except AttributeError:
            pass






#__all__ = [
#"bow_bert",
#"shuffle_bert",
#"sort_bert",
#"longformer",
#"longformer-qa",
##"bigbert",
#"crossencoder",
##"idcm",
##"nboost_crossencoder",
##"crossencoder_2",
##"electra",
##"roberta.shuffle",
##"bert",
##"bert_tf",
#"minilm6",
#"minilm12",
##"tinybert",
##"duobert",
##"contriever",
##"tctcolbert",
##"sparse_bert",
##"splade",
##"monolarge",
##"cocondenser",
##"tasb",
##"distilldot",
##"sentencebert"
#]

