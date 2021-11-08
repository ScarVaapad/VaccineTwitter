import pandas as pd
import os
import re
import gensim
from collections import defaultdict
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer
import logging
import numpy as np
import tqdm

df = pd.read_csv(os.path.join('data',"hydrated_clean_1.csv"))

# Transfer the time data from String to datetime and compare it our selected time
df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(None)

# Generating corpus from all the texts, and remove 'rt' for retweets
texts = [[word for word in document.split() if word not in ['rt','vaccine','covid19','coronavirus','covid-19','vaccination','covid','&amp','-']] for document in df.tweet_wordbag]
    # Possible removing the less frequent words(like used only once)
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1

texts =[[token for token in text if frequency[token]>1] for text in texts]

dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = gensim.models.LdaMulticore(corpus,id2word=dictionary, num_topics=10)
coherence_model_lda = CoherenceModel(model=lda_model,texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda=coherence_model_lda.get_coherence()

print("Coherence "+str(coherence_lda))
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#supporting function for hyperparameter tuning
def compute_coherence_values(corpus,dictionary,k,a,b):
    lda_model=gensim.models.LdaMulticore(corpus=corpus,
                                         id2word=dictionary,
                                         num_topics=k,
                                         random_state=100,
                                         chunksize=100,
                                         passes=10,
                                         alpha=a,
                                         eta=b)
    coherence_model_lda = CoherenceModel(model=lda_model,texts=texts, dictionary=dictionary,coherence='c_v')

    return coherence_model_lda.get_coherence()

grid = {}
grid['Validation_Set']= {}

# Topic range
min_topics = 5
max_topics = 20
step_size=1
topics_range = range(min_topics,max_topics,step_size)

# Alpha parameter
alpha = list(np.arange(0.01,1,0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

# Validation sets
num_of_docs = len(corpus)
corpus_sets = [gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)),
               corpus]

corpus_title = ['75% Corpus', '100% Corpus']

model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                 }

# Can take a long time to run
if 1==1:
    pbar = tqdm.tqdm(total=(len(beta)*len(alpha)*len(topics_range)*len(corpus_title)))

    #iterate through validation corpuses
    for i in range(len(corpus_sets)):
        for k in topics_range:
            for a in alpha:
                for b in beta:
                    cv = compute_coherence_values(corpus=corpus_sets[i],dictionary=dictionary,k=k,a=a,b=b)

                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)

                    pbar.update(1)


    pd.DataFrame(model_results).to_csv(os.path.join('models','lda_tuning_results.csv'),index=False)
    pbar.close()

# Models save & load
# model.save(os.path.join('models','all-20-lda.lda'))
# lda_model = gensim.models.LdaModel.load(os.path.join('models','all-20-lda.lda'))

# Print the model
# for i in range(0,model.num_topics-1):
#     print(model.print_topic(i))
# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))

df[(df['created_at'] > pd.Timestamp(2020,9,28,0)) & (df['created_at'] <= pd.Timestamp(2020,9,30,0))]

for document in df.tweet_wordbag:
    print(document)
# import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from nltk.stem.porter import *
# import numpy as np
# np.random.seed(2018)
# import nltk
# nltk.download('wordnet')
#
# def lemmatize_stemming(text):
#     stemmer = PorterStemmer()
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# def preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             result.append(lemmatize_stemming(token))
#     return result
#
# processed_docs = df['text'].map(preprocess)
# pdd = df['tweet_wordbag'].map(preprocess)
# processed_docs[:10]
# pdd[:10]
#
# dictionary = gensim.corpora.Dictionary(pdd)
# dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
#
# bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
#
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
#
