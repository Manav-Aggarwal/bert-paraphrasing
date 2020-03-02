import json
import copy

def getJSONObj(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

original_train = getJSONObj('train-v2.0.json')
new_train_q = getJSONObj('baseline2/new_questions_train_fasttext.json')

original_dev = getJSONObj('dev-v2.0.json')
new_dev_q = getJSONObj('baseline2/new_questions_dev_fasttext.json')

modified_train = copy.deepcopy(original_train)
for topic in modified_train['data']:
        for paragraph in topic['paragraphs']:
            for qa in paragraph['qas']:
                id = qa['id']
                qa['question'] = new_train_q[id]

modified_dev = copy.deepcopy(original_dev)
for topic in modified_dev['data']:
        for paragraph in topic['paragraphs']:
            for qa in paragraph['qas']:
                id = qa['id']
                qa['question'] = new_dev_q[id]

mod_train_data = modified_train['data']
mod_dev_data = modified_dev['data']
train_dev_combined = {}
train_dev_combined["version"] = "v2.0"
train_dev_combined["data"] = mod_train_data + mod_dev_data

with open('baseline2/modified_train_fasttext.json', 'w') as fp:
    json.dump(modified_train, fp)

with open('baseline2/modified_dev_fasttext.json', 'w') as fp:
    json.dump(modified_dev, fp)

with open('baseline2/train_dev_combined_fasttext.json', 'w') as fp:
    json.dump(train_dev_combined, fp)