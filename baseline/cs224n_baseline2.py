import json
import spacy
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from gensim.models.wrappers import FastText
# from gensim.models import FastText
model = FastText.load_fasttext_format('wiki.simple')
print("Model Loaded...")
nlp = spacy.load("en_core_web_sm")

def isPunctuation(token):
    return token.text.lower() in string.punctuation

def isStopWord(token):
    return token.is_stop

def isInContext(token, context_tokens_lemmas):
    return token.lemma_.lower() in context_tokens_lemmas

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

q2nq = {}
flag = False
with open('dev-v2.0.json') as json_file:
    data = json.load(json_file)
    count = 0
    datalen = len(data['data'])
    for topic in data['data']:
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
                id = qa['id']
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
                q2nq[id] = new_question
                i += 1
        count += 1
        print("{}/{} topics done".format(count, datalen))

with open('new_questions_dev_fasttext.json', 'w') as fp:
    json.dump(q2nq, fp, sort_keys=True)

