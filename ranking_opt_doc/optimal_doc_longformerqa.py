from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
import torch
import json
from tqdm import tqdm

def get_scores(model, tokenizer, queries, docs, bz=80):
    scores = list()
    print(len(queries))
    with torch.no_grad():
        for i in tqdm(range(0,len(queries), bz)):
            max_ind = min(len(queries), i+bz)
            score = model(**tokenizer(queries[i:max_ind], docs[i:max_ind], padding=True, truncation=True, return_tensors='pt').to('cuda')).logits[:, 1]
            scores.append(score)
    return torch.concat(scores)
    #return scores.logits.ravel()

def get_score(model, tokenizer, query, doc):
     return model(**tokenizer([query], [doc], padding=True, truncation=True, return_tensors='pt').to('cuda')).logits[:, 1].ravel().item()

def add_token(doc, doc_tokens):
    doc_inputs = list()
    for token in doc_tokens:
        doc_inputs.append(f"{doc} {token}")
    return doc_inputs

def remove_token(doc_tokens):
    doc_inputs = list()
    for i, token in enumerate(doc_tokens):
        tmp = doc_tokens.copy()
        del tmp[i]
        doc_inputs.append(' '.join(tmp))
    return doc_inputs

#doc = "cuba cuban sweet desserts dessert fruits fruit tropical spanish leche rum pudding coconut eaten made sugar boniatillo churros dulce flavours custard cinnamon citrus hint bread sauce cream ice popular de enforex cubans caramel guava vanilla > abundance cake spain mango drinks wine dish apple recipe percentage style commonly tooth natural english something most marmelades anonaceae cinammon wherby pudin diplomatico torrejas panatela borracha cristianos marmalades natillas moros creaminess nispero jerez eggy mamoncillo quesillo mouthwatering doused flan marmalade arroz cherimoya unsubscribe plantain tangerine yams creme plantains accompaniment doughnut mojo evaporated jams sherry pastries funnel found dipped cheesecake drunken pastry delight flavour fillings compliment milky diplomatic originating accompany toast cakes avocado pineapple mamey papaya bananas staple sunshine crispy sponge banana lime integral sour whilst drunk generations surprise varieties small stays closest newsletter con gained chocolate lemon truly courses butter breakfast whereas shell include shaped unlike prepared speaking equivalent varies fried thick juice orange typical remains served thin liquid warm rice offers ones description potatoes type fresh covered extra described deep quite pounds y starting throughout milk consider itself options countries content hot french email guide instead version food diet course healthy makes try similar site study search sometimes light available which vitamin white point those using often under used part us being both best its any like good see cause very some this are with their is then them have c has such long a many would or they been ' not out use an and ! these so in who no of that for about do other as , you all one may it from but be 2 ( ) the if more s : at to can . by / - how edit text pdf file acrobat xi xilearn files intuitive click functionality donna baker – october 2012in tutorial learn new editing features make easy when don t original rearrange paragraphs crop visuals streamlined tools view transcript xidonna 2012how i change my ? just follow 6 simple steps your std pro & images tool configure open each object shows outline select item format activate pane find block shown list contents box delete press enter space add on page drag draw paste clipboard appearance font drop down arrow modify underline frame adjust final position check typos color swatch pick horizontal scaling value increase character width spacing between characters line lines location layout edge paragraph alignment wider center centers within ’ separate number boxes move changes replace feature ctrl + f windows command mac want pop up opens replacement next first instance term highlights replaced continue process ok finish share looking help consult interactive ask question our forum leave comment below author products xirelated topics pdfs top searches convert word excel power point114 comments now closed deborah8 2016 03 10 2016not certain was answered noticed am adding usually escape button will take me mode thing remember sometime what image actual pretty evident ; clicking lori kassuba2 02 2016hi janina kowalska making sure language set polish properties advanced tab reading thanks kowalska4 18 2016i unsuccessfully trying adobe fonts bookman old group embedded grateful advice kind regards"

doc = ". Cuba Guide > About Cuba > Food in Cuba > Cuban Desserts Food Desserts Moros y Cristianos Mojo Sauce Dulce de Leche Boniatillo Cubans have a natural sweet tooth. Their rum is made from sugar and they have an abundance of tropical fruits available so it is no surprise that Cuban desserts and drinks are a delight for those of us who have a sweet tooth. Starting with the healthy and natural options, most of Cuba's tropical fruit stays in Cuba and is eaten fresh. Fruit is and has been an integral part of the Cuban diet for many generations, to the point that plantains and bananas are such a staple that the Cubans do not consider them as fruit at all. A small percentage of fruit is used for fruit juice drinks or made in to jams and marmelades, but this is a small percentage. Tropical fruits which can be found in Cuba include: anonaceae (type of sugar apple), avocado, banana, cherimoya (custard apple), coconut, guava, lemon, lime, mamey, mamoncillo, mango, nispero, orange (both sweet and sour varieties) papaya, pineapple, plantain and tangerine. The abundance of sugar, rum and fruits makes Cuban desserts some of the most mouthwatering and the ones which have fruit in are of course good for you with their vitamin C content and not the cause of any extra pounds gained whilst in Cuba!Some of the typical Cuban desserts you may try out include:-Arroz con Leche This is a rice pudding with a hint of citrus and cinnamon flavours which compliment the sweet creaminess of the liquid sauce Boniatillo A pudding made of sweet potatoes or yams. See Boniatillo recipe Churros Similar to Spanish churros, these are long thin crispy (deep fried) doughnut type pastries covered in sugar and eaten dipped in a thick hot chocolate sauce, which is then drunk if any remains. In Spain churros are a breakfast dish, whereas in Cuba they are eaten for dessert Dulce de Leche A popular dish throughout Spanish speaking countries, this is a very sweet milky dessert with a light caramel flavour. See Dulce de Leche recipe Flan Crème Caramel is the English equivalent to this dessert. Originating from Spain the Cuban version sometimes varies for its use of coconut, cinnamon or citrus flavours instead of vanilla Ice Cream Under the warm sunshine ice cream is a popular dessert which is prepared using tropical fruits. Coconut ice cream served in the coconut's shell is quite commonly found Marmalade Made with one or more tropical fruits, guava and mango being 2 of the most popular flavours, Cuban marmalades often accompany other desserts or are used as dessert or pastry fillings Natillas A sweet vanilla custard that has a hint of citrus and cinammon. Unlike English custard it is eaten by itself and not an accompaniment to a sponge cake Panatela Borracha A truly 'drunken cake' wherby the funnel shaped cakes are doused with rum, sweet wine/sherry (something like Spanish Jerez)Pudín Diplomático The closest description to 'Diplomatic Bread Pudding' would be that it something like bread and butter pudding with its fruits, rum and hint of cinnamon Quesillo This is a cheesecake style dessert Torrejas Most commonly described as french toast Cuban style, this eggy bread is made with white wine and evaporated milk Site Search Enforex Cuba Study Spanish in Cuba Enforex offers you the best Spanish courses in Cuba. Newsletter Email:- Unsubscribe"
query = 'what is mamey'

#doc = 'From U.S. Citizenship and Immigration Services (USCIS) Naturalization Guide to Naturalization Child Citizenship Act Naturalization Test. Laws and Regulations Read the Code of Federal Regulation Chapter 8 Section 319.2, Expeditious Naturalization regulation and read the INA section 319(b). Department of State Employees and Spouses Only'

#query = 'are naturalization records public information'
#model = AutoModelForSequenceClassification.from_pretrained('/project/draugpu/experiments_ictir/bert/bz_128_lr_3e-06/model_30/')
#model = AutoModelForSequenceClassification.from_pretrained('dmrau/bow-bert')
#model = AutoModelForSequenceClassification.from_pretrained('dmrau/crossencoder-msmarco')
model = AutoModelForSequenceClassification.from_pretrained("dmrau/longformer-qa-msmarco")
tokenizer = AutoTokenizer.from_pretrained("aware-ai/longformer-QA", use_auth_token=True)
model = model.to('cuda')
model.eval()






doc_tokens = tokenizer.tokenize(doc)[:2000]
remaining_tokens = doc_tokens.copy()
optim_doc = ""
if False:
    print('optimal_doc')
    for i in range(len(remaining_tokens)):
        batch_docs = add_token(optim_doc, remaining_tokens)
        scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
        max_idx = torch.argmax(scores)
        optim_term  = remaining_tokens[max_idx]
        optim_doc += optim_term + ' '
        print(scores[max_idx].item(), optim_doc)
        remaining_tokens.remove(optim_term)
if True:
    print('remove terms')
    remaining_tokens = doc_tokens.copy()
    #remaining_tokens = tokenizer.tokenize(optim_doc)[:64]
    batch_docs = remove_token(remaining_tokens)
    scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
    _, indices = torch.topk(scores.ravel(), 64)
    new_doc = ''
    print('doc single:', " ".join(remaining_tokens))
    for ind in indices:
        new_doc += ' ' + doc_tokens[ind]
    score = get_score(model, tokenizer, query, new_doc)
    print('score:', round(score, 2))

exit()
print('remove terms')
remaining_tokens = doc_tokens.copy()
print(remaining_tokens)
for i in range(len(remaining_tokens)):
    batch_docs = remove_token(remaining_tokens)
    scores = get_scores(model, tokenizer, [query] * len(batch_docs) , batch_docs) 
    max_idx = torch.argmax(scores)
    for token, score in zip(remaining_tokens, scores):
        print(token, round(score.item(), 2))
    #print('removing', remaining_tokens[max_idx])
    del remaining_tokens[max_idx]
    #print(scores[max_idx].item(), ' '.join(remaining_tokens))
    if len(remaining_tokens) <= 64:
        break
print('doc iteratively', " ".join(remaining_tokens))
score = get_score(model, tokenizer, query, " ".join(remaining_tokens))
print('score:', round(score, 2))



js = json.loads('{"id": "1020327", "contents": "From U.S. Citizenship and Immigration Services (USCIS) Naturalization Guide to Naturalization Child Citizenship Act Naturalization Test. Laws and Regulations Read the Code of Federal Regulation Chapter 8 Section 319.2, Expeditious Naturalization regulation and read the INA section 319(b). Department of State Employees and Spouses Only", "vector": {"from": 16, "u": 32, ".": 51, "s": 10, "citizenship": 132, "and": 16, "immigration": 91, "services": 37, "(": 17, "usc": 107, "##is": 55, ")": 15, "natural": 144, "##ization": 80, "guide": 93, "to": 47, "child": 92, "act": 87, "test": 90, "laws": 62, "regulations": 70, "read": 55, "the": 25, "code": 103, "of": 4, "federal": 80, "regulation": 75, "chapter": 78, "8": 94, "section": 74, "319": 122, "2": 33, ",": 14, "ex": 57, "##ped": 75, "##iti": 64, "##ous": 33, "ina": 109, "b": 36, "department": 87, "state": 64, "employees": 56, "spouse": 86, "##s": 22, "only": 81, "[SEP]": 0}}')

doc_splade = sorted(js['vector'], key=js['vector'].get, reverse=True)
doc_splade = " ".join(doc_splade[:32])
print('doc splade:', doc_splade)
score = get_score(model, tokenizer, query, doc_splade)
print('score:', round(score, 2))


