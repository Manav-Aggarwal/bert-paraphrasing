import json
import copy
import random

def getJSONObj(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

combined_dataset = getJSONObj('baseline2/train_dev_combined_fasttext.json')
data = combined_dataset['data']
random.shuffle(data)

total = len(data)
train_amt  = int(.7 * total)
dev_amt = int(.2 * total)

combined_train = {'version':'v2.0'}
combined_train['data'] = data[:train_amt]

combined_dev = {'version':'v2.0'}
combined_dev['data'] = data[train_amt: train_amt+dev_amt]

combined_test = {'version': 'v2.0'}
combined_test['data'] = data[train_amt+dev_amt:]

with open('baseline2/new_dataset/new_split_train.json', 'w') as fp:
    json.dump(combined_train, fp)

with open('baseline2/new_dataset/new_split_dev.json', 'w') as fp:
    json.dump(combined_dev, fp)

with open('baseline2/new_dataset/new_split_test.json', 'w') as fp:
    json.dump(combined_test, fp)

