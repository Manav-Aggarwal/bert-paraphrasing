import json
import spacy
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
nlp = spacy.load("en_core_web_sm")

def isStopWord(token):
    return token.is_stop

def isInContext(token, context_tokens):
    return token.text.lower() in context_tokens

def isPunctuation(token):
    return token.text in string.punctuation

q2nq = {}

flag = False
with open('train-v2.0.json') as json_file:
    data = json.load(json_file)
    for topic in data['data']:
        # if flag:
        #     break
        for paragraph in topic['paragraphs']:
            # if flag:
            #     break
            new_questions = []
            context = paragraph['context']
            context_tokens = [t.text.lower() for t in nlp(context)]
            i = 0
            for qa in paragraph['qas']:
                # if i > 5:
                #     flag = True
                id = qa['id']
                question = qa['question']
                question_doc = nlp(question)
                new_question_tokens = []
                for token in question_doc:
                    if isPunctuation(token) or isStopWord(token) or isInContext(token, context_tokens):
                        new_question_tokens.append(token.text)
                new_question = TreebankWordDetokenizer().detokenize(new_question_tokens)
                q2nq[id] = new_question
                i += 1

with open('new_questions_train.json', 'w') as fp:
    json.dump(q2nq, fp)
