#!/usr/bin/env python
# coding: utf-8

# In[6]:


#from keras.datasets import imdb
#from keras import preprocessing
import numpy as np
import os
import pandas as pd
import re
import random 
from nltk.tokenize import sent_tokenize
#from nltk.corpus import stopwords


# In[3]:


## Read corpus
fnames = []
titles = []
authors = []
#stop_words = set(stopwords.words('english'))

for file in os.listdir():
    ext = os.path.splitext(file)[1]
    if ext == '.txt':
        #print(os.path.splitext(file)[0])
        fnames.append(file)
        title = os.path.splitext(file)[0].split('_')[0]
        author = os.path.splitext(file)[0].split('_')[1]
        titles.append(title)
        authors.append(author)


# In[60]:


#punctuation = re.sub(r"([^\d+])(\.|!|\?|;|\n|。|！|？|；|…|　|!|؟|؛)+")
def clean_lines(fname, title_or_author = 'titles'):
    global dic0
    dic0 = {eval(title_or_author)[0]:0,
           eval(title_or_author)[1]:1,
           eval(title_or_author)[2]:2}
    with open(fname,encoding='utf8') as f:
        tokens = sent_tokenize(f.read())
        sentences= []
        labels = []
        for sent in tokens:
            sent = re.sub("[^A-Za-z]", " ", sent.strip()).lower()
            #sent = re.sub(r"([^\d+])(\.|!|\?|;|\n|。|！|？|；|…|　|!|؟|؛)+",'',sent)
            #sent = re.sub(r'\d+', '',sent)
            sentences.append(sent)
            index = fnames.index(fname)
            #print(index)
            labels.append(dic0[eval(title_or_author)[index]])
    return(sentences,labels)


# In[61]:


f1,lab1 = clean_lines(fnames[0])
f2,lab2 = clean_lines(fnames[1])
f3,lab3 = clean_lines(fnames[2])


# In[54]:


## Store the label, sentence in dataframe 
df = pd.DataFrame(columns = ['sentence','label'])
df.sentence = f1+f2+f3
df.label = lab1+lab2+lab3


# In[56]:


## Split the one_hot_results and its corresponding labels to train, test
def random_cut(df,num_max=None):
    if num_max != None:
        indx = random.sample(range(len(df)),num_max)
        df = df.iloc[indx,:].reset_index()
    #print(len(df))
        
    cut_test = random.sample(range(len(df)),round(len(df)/5)) # 20% test
    comp_cut = [i for i in range(len(df)) if i not in cut_test] 
    cut_val = random.sample(comp_cut,round(len(comp_cut)/5)) # 16% val
    cut_train = [i for i in range(len(df))                  if i not in cut_val and i not in cut_test]# 64% train
    print(len(cut_train))
    print(len(cut_test))
    print(len(cut_val))
    
    df.loc[cut_train,'type'] = 'train'
    df.loc[cut_test,'type'] = 'test'
    df.loc[cut_val,'type'] = 'val'
    return(df)


# In[62]:


df = random_cut(df)
print(dic0)


# In[58]:


df.to_csv('processed_data .csv')


# In[64]:


### RUN this only in jupyter notebook
os.system(f'jupyter nbconvert 01-clean.ipynb --to python')


# In[ ]:




