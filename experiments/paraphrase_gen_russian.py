import torch
import json
import copy

# Round-trip translations between English and German:
en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

def getJSONObj(filename):
    with open(filename) as f:
        data = json.load(f)
        return data

modified_squad_train = getJSONObj('modified_squad_train.json')
modified_squad_dev = getJSONObj('modified_squad_dev.json')
modified_squad_test = getJSONObj('modified_squad_test.json')

def paraphrase(text):
    return ru2en.translate(en2ru.translate(text))

def paraphrase_dataset(dataset):
    count = 0
    datalen = len(dataset['data'])
    for topic in dataset['data']:
        for paragraph in topic['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                paraphrased_question = paraphrase(question)
                qa['question'] = paraphrased_question
        count += 1
        print("{}/{} topics done".format(count, datalen))
    return dataset

russian_squad_train = paraphrase_dataset(modified_squad_train)
with open('russian_squad_train.json', 'w') as f:
    json.dumps(paraphrased_squad_train, f)

russian_squad_dev = paraphrase_dataset(modified_squad_dev)
with open('russian_squad_dev.json', 'w') as f:
    json.dumps(paraphrased_squad_dev, f)

russian_squad_test = paraphrase_dataset(modified_squad_test)
with open('russian_squad_test.json', 'w') as f:
    json.dumps(paraphrased_squad_test, f)

paraphrased_squad_train = paraphrase_dataset(modified_squad_train)
paraphrased_squad_dev = paraphrase_dataset(modified_squad_dev)
paraphrased_squad_test = paraphrase_dataset(modified_squad_test)
