"""import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
text=''
def summarizer(rawdoc):
    
    stopwords= list(STOP_WORDS)
    #print(stopwords)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdoc)
    #print(doc)
    tokens = [token.text for token in doc]
    #print(tokens)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text]=1
            else:
                word_freq[word.text]+=1
    #print(word_freq)
    max_freq=max(word_freq.values())
    #print(max_freq)
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
    #print(word_freq)

    sent_tokens = [sent for sent in doc.sents]
    #print(sent_token)

    sent_scores ={}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    #print(sent_scores)

    select_len = int(len(sent_tokens)*0.5)
    #print(select_len)

    summary = nlargest(select_len,sent_scores,key=sent_scores.get)
    #print(summary)
    final_summary = [word.text for word in summary]
    summary=' '.join(final_summary)
    #print(text)
    #print()
    #print(summary)
    #print(" Words in original text: ",len(text.split()))
    #print(" Words in summary text: ",len(summary.split()))
    
    return summary,doc,len(rawdoc.split(' ')),len(summary.split(' '))"""


"""import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import itertools

def summarizer(rawdoc, summary_ratio=0.5, metric='freq'):
    
    # Load spaCy model and create stopword list
    nlp = spacy.load('en_core_web_sm')
    stopwords= list(STOP_WORDS)
    
    # Tokenize the input text
    doc = nlp(rawdoc)
    
    # Calculate word frequencies and sentence scores
    word_freq = {}
    sentence_scores = {}
    
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
                if word.text not in word_freq:
                    word_freq[word.text] = 1
                else:
                    word_freq[word.text] += 1
                    
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_freq[word.text]
                else:
                    sentence_scores[sent] += word_freq[word.text]
    
    # Normalize word frequencies
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq
        
    # Normalize sentence scores
    if metric == 'freq':
        max_score = max(sentence_scores.values())
        for sent in sentence_scores:
            sentence_scores[sent] = sentence_scores[sent] / max_score
    elif metric == 'pos':
        max_score = len(list(doc.sents))
        for i, sent in enumerate(doc.sents):
            sentence_scores[sent] = 1 - (i / max_score)
    elif metric == 'length':
        max_score = max([len(sent) for sent in doc.sents])
        for sent in sentence_scores:
            sentence_scores[sent] = len(sent) / max_score
    else:
        for sent1, sent2 in itertools.combinations(doc.sents, 2):
            sim = sent1.similarity(sent2)
            if sent1 not in sentence_scores:
                sentence_scores[sent1] = sim
            else:
                sentence_scores[sent1] += sim
            if sent2 not in sentence_scores:
                sentence_scores[sent2] = sim
            else:
                sentence_scores[sent2] += sim
        
        max_score = max(sentence_scores.values())
        for sent in sentence_scores:
            sentence_scores[sent] = sentence_scores[sent] / max_score
    
    # Select top sentences based on score
    num_sentences = max(1, int(len(list(doc.sents)) * summary_ratio))
    summary = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Concatenate top sentences into summary text
    final_summary = [sent.text.strip() for sent in summary]
    summary = ' '.join(final_summary)
    
    # Return summary text and summary statistics
    return summary,doc,len(rawdoc.split()), len(summary.split())"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from collections import Counter, defaultdict

def summarizer(rawdoc):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(rawdoc)
    
    stopwords = list(STOP_WORDS)
    word_freq = Counter(token.text.lower() for token in doc
                         if token.text.lower() not in stopwords and
                            token.text.lower() not in punctuation)
    max_freq = max(word_freq.values())
    word_freq = {word: freq/max_freq for word, freq in word_freq.items()}

    sent_scores = defaultdict(float)
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_scores[sent] += word_freq[word.text.lower()]

    summary_len = int(len(list(doc.sents)) * 0.5)
    summary = nlargest(summary_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for sent in summary for word in sent]
    summary = ' '.join(final_summary)
    keywords = [word.text for word in doc if word.text.lower() not in stopwords and
                word.text.lower() not in punctuation and
                word.text not in summary]
    keyword_freq = Counter(keywords)
    #top_keywords = [keyword for keyword, freq in keyword_freq.most_common(10)]
    top_keywords = ', '.join([keyword for keyword, freq in keyword_freq.most_common(10)])


    return summary, doc, len(doc), len(summary.split()),top_keywords



