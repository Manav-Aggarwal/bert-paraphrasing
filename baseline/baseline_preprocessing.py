import json
import string
import random

def getJSONObj(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

# opening original SQuAD 2.0 train and dev data
squad_train = getJSONObj('original_squad_dataset/train-v2.0.json')
squad_dev = getJSONObj('original_squad_dataset/dev-v2.0.json')

# combining SQuAD 2.0 train and dev data and then shuffling
squad_train_dev_combined = {"version": "v2.0"}
squad_train_dev_combined['data'] = squad_train['data'] + squad_dev['data']
random.shuffle(squad_train_dev_combined['data'])

import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
import copy
nlp = spacy.load("en_core_web_sm")

def isPunctuation(token):
    return token.text.lower() in string.punctuation

def isStopWord(token):
    return token.is_stop

def isInContext(token, context_token_lemmas):
    return token.lemma_.lower() in context_token_lemmas

b1_train_dev_combined = copy.deepcopy(squad_train_dev_combined)

flag = False
for topic in b1_train_dev_combined['data']:
    # if flag:
    #     break
    for paragraph in topic['paragraphs']:
        # if flag:
        #     break
        new_questions = []
        context = paragraph['context']
        context_tokens = [t for t in nlp(context) if not isPunctuation(t) and not isStopWord(t)]
        context_token_text = [t.text for t in context_tokens]
        context_token_lemmas = [t.lemma_.lower() for t in context_tokens]
        i = 0
        while i < len(paragraph['qas']):
            question = paragraph['qas'][i]['question']
            question_doc = nlp(question)
            new_question_tokens = []
            for token in question_doc:
                if isPunctuation(token) or isStopWord(token) or isInContext(token, context_token_lemmas):
                    new_question_tokens.append(token.text)
            new_question = TreebankWordDetokenizer().detokenize(new_question_tokens)
            if new_question == "":
                paragraph['qas'].pop(i)
            else:
                paragraph['qas'][i]['question'] = new_question
                i += 1

from gensim.models.wrappers import FastText
model = FastText.load_fasttext_format('wiki.simple')
print("Model Loaded...")
nlp = spacy.load("en_core_web_sm")

def getMostSimContextWord(token, context_tokens):
    highest_sim = float('-Inf')
    most_sim_word = None
    for ct in context_tokens:
        # if isPunctuation(ct) or isStopWord(ct):
        #     continue
        # else:
        if ct.lower() in model.wv.vocab:
            curr_sim = model.similarity(token.text.lower(), ct.lower())
            if curr_sim >= highest_sim:
                highest_sim = curr_sim
                most_sim_word = ct
    return most_sim_word

b2_train_dev_combined = copy.deepcopy(squad_train_dev_combined)

flag = False
count = 0
datalen = len(b2_train_dev_combined['data'])
for topic in b2_train_dev_combined['data']:
    # if flag:
    #     break
    for paragraph in topic['paragraphs']:
        # if flag:
        #     break
        context = paragraph['context']
        context_tokens = [t for t in nlp(context) if not isPunctuation(t) and not isStopWord(t)]
        context_token_text = [t.text for t in context_tokens]
        context_token_lemmas = [t.lemma_.lower() for t in context_tokens]
        i = 0
        for qa in paragraph['qas']:
            # if i > 5:
            #     flag = True
            question = qa['question']
            question_doc = nlp(question)
            new_question_tokens = []
            for token in question_doc:
                if isPunctuation(token) or isStopWord(token) or isInContext(token, context_token_lemmas):
                    new_question_tokens.append(token.text)
                else:
                    if token.text.lower() in model.wv.vocab:
                        word = getMostSimContextWord(token, context_token_text)
                    else:
                        word = token.text
                    new_question_tokens.append(word)
            new_question = TreebankWordDetokenizer().detokenize(new_question_tokens)
            qa['question'] = new_question
            i += 1
    count += 1
    print("{}/{} topics done".format(count, datalen))

# Original SQuAD dataset
total_n = len(squad_train_dev_combined['data'])

train_n = int(.7 * total_n)
dev_n = int(.2 * total_n)

modified_squad_train = {'version':'v2.0'}
modified_squad_train['data'] = squad_train_dev_combined['data'][:train_n]

modified_squad_dev = {'version':'v2.0'}
modified_squad_dev['data'] = squad_train_dev_combined['data'][train_n: train_n+dev_n]

modified_squad_test = {'version': 'v2.0'}
modified_squad_test['data'] = squad_train_dev_combined['data'][train_n+dev_n: ]

with open('modified_squad_dataset/modified_squad_train.json', 'w') as fp:
    json.dump(modified_train, fp)

with open('modified_squad_dataset/modified_squad_dev.json', 'w') as fp:
    json.dump(modified_dev, fp)

with open('modified_squad_dataset/modified_squad_test.json', 'w') as fp:
    json.dump(modified_test, fp)

# Baseline 1 dataset
total_n = len(b1_train_dev_combined['data'])

train_n = int(.7 * total_n)
dev_n = int(.2 * total_n)

b1_train = {'version':'v2.0'}
b1_train['data'] = b1_train_dev_combined['data'][:train_n]

b1_dev = {'version':'v2.0'}
b1_dev['data'] = b1_train_dev_combined['data'][train_n: train_n+dev_n]

b1_test = {'version': 'v2.0'}
b1_test['data'] = b1_train_dev_combined['data'][train_n+dev_n: ]

with open('baseline1/b1_train.json', 'w') as fp:
    json.dump(b1_train, fp)

with open('baseline1/b1_dev.json', 'w') as fp:
    json.dump(b1_dev, fp)

with open('baseline1/b1_test.json', 'w') as fp:
    json.dump(b1_test, fp)

# Baseline 2 dataset
total_n = len(b2_train_dev_combined['data'])

train_n = int(.7 * total_n)
dev_n = int(.2 * total_n)

b2_train = {'version':'v2.0'}
b2_train['data'] = b2_train_dev_combined['data'][:train_n]

b2_dev = {'version':'v2.0'}
b2_dev['data'] = b2_train_dev_combined['data'][train_n: train_n+dev_n]

b2_test = {'version': 'v2.0'}
b2_test['data'] = b2_train_dev_combined['data'][train_n+dev_n: ]

with open('baseline2/b2_train.json', 'w') as fp:
    json.dump(b2_train, fp)

with open('baseline2/b2_dev.json', 'w') as fp:
    json.dump(b2_dev, fp)

with open('baseline2/b2_test.json', 'w') as fp:
    json.dump(b2_test, fp)