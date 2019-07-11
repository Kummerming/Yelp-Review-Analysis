import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from multiprocessing import Pool
import re
import time
from nltk.corpus import stopwords

# use parallel computing to perform data cleaning


def wordnet_pos(x):                                # a function to denote position of speech for lemmatization
    if x.startswith('V'):
        return wordnet.VERB
    else:
        return wordnet.NOUN


def sent_tokenize(x):
    # this function is the main tokenizer for data cleaning
    stopword = set(stopwords.words('english')) - {'he', 'him', 'his', 'himself', 'not', 'no', 'nor',
                                                  'she', 'her', "she's", 'her', 'hers', 'herself',
                                                  'they', 'them', 'their', 'theirs', 'themselves'}

    lmtzer = WordNetLemmatizer()
    # tokenizer = RegexpTokenizer(r'\w+')
    x = x.lower()
    temp = re.sub(",", '.', x)
    temp = re.sub('n\'t', ' not', temp)
    word = re.findall('[a-zA-Z]+|:\)|\.\.\.+|[!]+|\!\?|\.', temp)
    word = [i for i in word if i not in stopword]           # delete stopwords
    word = mark_negation(word)

    word_tag = nltk.pos_tag(word)                   # lemmatization, very time consuming
    lmt_word = [lmtzer.lemmatize(i_pair[0], pos=wordnet_pos(i_pair[1])) for i_pair in word_tag]
    lmt_word = " ".join(lmt_word)          # combine with space, easy for other use
    return lmt_word


def multi_rev(data):
    data.text = data.text.apply(sent_tokenize)
    return data


num_cores = 12  # number of cpu cores


def parallelize_dataframe(df, func):        # the preparation for parallel processing
    df = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__':          # perform parallel cleaning process
    for i in range(9):                  # slice the original data into 9 pieces use linux shell to save RAM
        rev_data = pd.read_json(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev0' + str(i), lines=True,
                                orient='records')
        print('done reading ' + str(i))

        print('start cleaning ' + str(i))
        start = time.time()
        rev_data = parallelize_dataframe(rev_data, multi_rev)
        end = time.time()
        print('done ' + str(i))
        print(end - start)
        rev_data.to_csv(r'D:\OneDrive - UW-Madison\Module2\Data_Module2\rev_clean' + str(i) + '.csv', index=False)
        del rev_data






