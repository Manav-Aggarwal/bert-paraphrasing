import torch
import json
import copy

# Round-trip translations between English and German:
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

def getJSONObj(filename):
    with open(filename) as f:
        data = json.load(f)
        return data

modified_squad_train = getJSONObj('modified_squad_train.json')
modified_squad_dev = getJSONObj('modified_squad_dev.json')
modified_squad_test = getJSONObj('modified_squad_test.json')

def paraphrase(text):
    return de2en.translate(en2de.translate(text))

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


paraphrased_squad_train = paraphrase_dataset(modified_squad_train)
with open('german_squad_train.json', 'w') as f:
    json.dumps(paraphrased_squad_train, f)

paraphrased_squad_dev = paraphrase_dataset(modified_squad_dev)
with open('german_squad_dev.json', 'w') as f:
    json.dumps(paraphrased_squad_dev, f)

paraphrased_squad_test = paraphrase_dataset(modified_squad_test)
with open('german_squad_test.json', 'w') as f:
    json.dumps(paraphrased_squad_test, f)