# Import libs
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import  Counter
from sklearn.feature_extraction.text import CountVectorizer
from urllib.request import urlopen
import string, random, re
from PIL import Image
from wordcloud import WordCloud
from afinn import Afinn


# Defined style for almost whole project at one place
hist_kwargs = dict(bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

# Dictionary with Author names for more understandable Plots
author_dict = {
    'EAP': 'Edgar Allan Poe',
    'HPL': 'H. P. Lovecraft',
    'MWS': 'Mary Shelley'
    }

# Code Snippet for Initial Analysis
def initial_insight (data):       
    # Initial data analysis
    size = data.shape
    sum_duplicates = data.duplicated().sum()
    sum_null = data.isnull().sum().sum()
    is_nan = data.isnull()
    row_has_nan = is_nan.any(axis=1)
    rows_with_nan = data[row_has_nan]
    count_nan_rows = rows_with_nan.shape
    
    data['full_name'] = data['author'].map(author_dict.get) 
    
    print("Number of Samples: %d,\nNumber of Features: %d,\nDuplicated Entries: %d,\nNull Entries: %d,\
          \nNumber of Rows with Null Entries: %d %.1f%%"%(size[0], size[1], sum_duplicates, 
          sum_null, count_nan_rows[0], (count_nan_rows[0]/data.shape[0])*100))
    
    plot_author_story_number_relation(data, 'full_name', 'Author','Count', 'Samples per Author')

# Code Snippet for Plotting Author - # of Stories Relation
def plot_author_story_number_relation(data, x, x_axis_title,y_axys_title, plot_title):
    plt.figure(figsize=[value for key, value in hist_kwargs.items() if key=='figsize'][0])
    sns.set(style="ticks", font_scale=1)
    ax = sns.countplot(data=data, x=x, order=data[x].value_counts().index, palette="Greens_r")
    for p in ax.patches:
        ax.annotate("%d"%p.get_height(), (p.get_x()+p.get_width()/2., abs(p.get_height())),
                    ha='center', va='bottom', color='black', xytext=(0, 3), rotation='horizontal', textcoords='offset points')
        
    sns.despine(top=True, right=True, left=True, bottom=False)
    plt.xticks(rotation=0, fontsize=12)
    ax.set_xlabel(x_axis_title, fontsize=14, weight='bold')
    ax.set(yticklabels=[])
    ax.axes.get_yaxis().set_visible(False)  
    plt.title(plot_title, fontsize=16, weight='bold')
    
# Code Snippet for Character Length per Excerpt Histogram
def plot_character_length_histogram(data):
    data.str.len().\
    hist(**hist_kwargs)
    
# Code Snippet for Number of Words per Excerpt Histogram
def plot_word_number_histogram(data):
    data.str.split().\
    map(lambda x: len(x)).\
    hist(**hist_kwargs)
       
# Code Snippet for Character Length per Word Histogram    
def plot_word_length_histogram(data):
    data.str.split().\
    apply(lambda x : [len(i) for i in x]).\
    map(lambda x: np.mean(x)).\
    hist(**hist_kwargs)
    
#NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords
language = 'english'
all_stopwords = set(stopwords.words(language))

# Code Snippet Corpus Creation
def create_stopwords_corpus(data):    
    corpus=[]
    texts = data.str.split()
    text = texts.values.tolist()
    corpus = [word for i in text for word in i]
    
    present_stopwords = defaultdict(int)
    for word in corpus:
        if word in all_stopwords:
            present_stopwords[word] += 1
            
    print("\n\nNumber of stopwords in present dataset: ", len([k for k in present_stopwords]), "\n\n")
    
    print(present_stopwords)
            
    return corpus, present_stopwords
            
# Code Snippet for Top Stopwords Barchart
def plot_top_stopwords_barchart(stopwords, num_top=10):
    top = sorted(stopwords.items(), key=lambda x:x[1],reverse=True)[:num_top] 
    x, y = zip(*top)
    plt.figure(figsize=(12,8))
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 
    plt.bar(x,y, color=[value for key, value in hist_kwargs.items() if key=='color'][0])

# Code Snippet for Top Non-Stopwords Barchart
def plot_top_non_stopwords_barchart(corpus, num_top=110): 
    counter = Counter(corpus)
    most = counter.most_common()
        
    x, y= [], []
    for word,count in most[:num_top]:
        if (word not in all_stopwords):
            x.append(word)
            y.append(count)
            
    plt.figure(figsize=(12,8))        
    sns.barplot(x=y, y=x)
    
# N-Grams Exploration
# Code Snippet for Top N-grams Barchart
def plot_top_ngrams_barchart(data, corpus, n=2, num_top=10):
    
    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:num_top]

    top_n_bigrams = _get_top_ngram(data,n)[:num_top]
    x, y = map(list, zip(*top_n_bigrams))
    sns.barplot(x=y, y=x)
    
# Word Frequency By Author
# Utility Functions to Create word Frequency
stem = PorterStemmer()

def process_list(text): #returns a list of preprocessed words    
    word_list = []
    #for t in text:           
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) #remove punctuations
    text = text.lower() #lower case
    tokenized_word = word_tokenize(text) #separate into words
    for word in tokenized_word:
        if word not in all_stopwords: #filter stop-words
            word = stem.stem(word) #stemming
            word_list.append(word) #append to general list
            
    return word_list
    
def build_freqs(texts, author):
    authorslist = np.squeeze(author).tolist()
    # Start with an empty dictionary and populate it by looping over all samples
    # and over all processed words in each sample.
    freqs = {}
    words_sample = []
    for text, author in zip(texts, authorslist):
        for word in process_list(text):
            words_sample.append(word)
            pair = (word, author)
            freqs[pair] = freqs.get(pair, 0) + 1  
            
    return freqs, words_sample

# Code Snippet for Word Frequency per Author Analysis
def word_freq_per_author_analysis(data): 
    freqs_author, words = build_freqs(data['text'], data['author'])

    freq_words = []

    for word in words:
        MWS = 0
        HPL = 0
        EAP = 0
        if (word, 'MWS') in freqs_author:
            MWS = freqs_author[(word, 'MWS')]
        if (word, 'HPL') in freqs_author:
            HPL = freqs_author[(word, 'HPL')]
        if (word, 'EAP') in freqs_author:
            EAP = freqs_author[(word, 'EAP')]      
        freq_words.append([word, MWS,HPL,EAP])   

    freq_words_df = pd.DataFrame(freq_words, columns = ['word', 'MWS','HPL','EAP'])    
    freq_words_df['sum'] =  freq_words_df.loc[:, ['MWS','HPL','EAP']].sum(axis=1)
    freq_words_df.sort_values('sum', ascending=False, inplace=True)
    freq_words_df.drop_duplicates(inplace=True)

    return freq_words_df
    
# Code Snippet for Plotting Word Frequency per Author    
def plot_word_freq_per_author(data):
    #get analysis for plotting
    freq_words_df = word_freq_per_author_analysis(data)
    
    z = 0; j = 0
    fig, axarr = plt.subplots(1, 3, figsize=(20, 5))

    authors_abbr = list(author_dict.keys())

    for author in authors_abbr:
        data = freq_words_df.loc[:,['word',author]]
        data.sort_values(author, ascending=False, inplace=True)
        ax = sns.lineplot(data=data[0:20], x="word", y=author, marker='o', ax=axarr[z])
        axarr[z].tick_params(axis='x', rotation=70)    
        axarr[z].set_xlabel('The 20 Most Common Words for '+author, fontsize=12, weight='bold')
        axarr[z].set_ylabel('Count', fontsize=12, weight='bold')
        axarr[z].set_title(author, fontsize=14, weight='bold');
        sns.despine(top=True, right=True, left=False, bottom=False)
        z += 1
        #print(FreqDF[0:15]['word'])

    fig.tight_layout(pad=3.0)
    plt.suptitle('Word Frequency per Author - Top 20 Words',fontsize=16, weight='bold');
    plt.show()

# Kernel Density Estimation
# Code Snippet for Creating a New Features
def create_new_features(data):
    punctuations = string.punctuation
    
    data['word_count'] = data['text'].apply(lambda x : len(x.split()))
    data['char_count'] = data['text'].apply(lambda x : len(x.replace(" ","")))
    data['word_density'] = data['word_count'] / (data['char_count'] + 1)

    #Adding +1 to allow ratio calculation
    data['uppercase'] = data['text'].str.findall(r'[A-Z]').str.len()+1
    data['lowercase'] = data['text'].str.findall(r'[a-z]').str.len()+1
    data['uppLow_ratio'] = data['uppercase'] / (data['lowercase'] + 1)

    data['punc_count'] = data['text'].apply(lambda x : len([a for a in x if a in punctuations]))

# Code Snippet for KDA Performing
def kde_analysis(data, features_list):    
    create_new_features(data)
    
#     features_list = ['word_count','char_count','word_density','uppercase','lowercase','uppLow_ratio','punc_count']
    
    for i in features_list:
        plt.figure(figsize=(20,5))
        ax = sns.kdeplot(data=data, x=i, linewidth=1, alpha=.3, fill=True, hue='full_name', palette='husl') 
        ax.set_xlabel(i)
        plt.title('KDE Plot - '+i, fontsize=16, weight='bold', pad=20);  
        sns.despine(top=True, right=True, left=False, bottom=False)
        
# Outliers
# Code Snippet for Box Plotting
def plot_box_and_whislers(data, features_list):
    fig, axarr = plt.subplots(1, 5, figsize=(20, 5))
    
    sns.set(style="ticks", font_scale=1)
    sns.despine(top=True, right=True, left=False, bottom=False)

#     features_list = ['word_count','char_count','word_density','UppLowRatio','punc_count']

    z = 0
    for j in range(0,5):
        ax = sns.boxplot(data=data, x='full_name', y=data[features_list[z]], ax=axarr[j], palette='husl')
        axarr[j].tick_params(axis='x', rotation=70)
        z +=1

    axarr[0].set_title("Number of Words per Author")
    axarr[1].set_title("Number of Characters per Author")
    axarr[2].set_title("Words and Characters Ratio per Author")
    axarr[3].set_title("Upper/Lower Case Letters Ratio per Author")
    axarr[4].set_title("Use of Punctuation per Author")

    fig.tight_layout(pad=3.0)
    plt.suptitle('Statistical Count Features',fontsize=16, weight='bold');

    plt.show()
        
# Cloud of Words
# Code Snippet for CoW per Author Visualisation
def word_cloud_per_instance(data):
    authors = list(author_dict.keys()) #'EAP', 'HPL', 'MWS'
    
    word_clouds = []        
    
    for index, author in enumerate(authors):
        # return random index of text per current author 
        sample_index = data.query("author == @author").sample(n=1).index[0]
        word_clouds.append(WordCloud().generate(data['text'][sample_index]))
        print("\n",data['text'][sample_index],"\n")
        print("\n",data['full_name'][sample_index],"\n")
        plt.imshow(word_clouds[index], interpolation='bilinear')
        plt.show()
    
# Code Snippet for CoW on Dataset Visualisation
def word_cloud_for_dataset(data):
    word_cloud = WordCloud(background_color="white", contour_width=0.1, colormap='YlGn', max_words=2000, max_font_size=50, random_state=42)
    text=" ".join(review for review in data['text'])
    word_cloud.generate(text)
    plt.figure(figsize=[20,10])
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()    
    
# Sentiment Analysis
def sentiment_analysis(data):    
    af = Afinn()

    # compute sentiment scores (polarity) and labels
    sentiment_scores = [af.score(text) for text in data['text']]
    sentiment_category = ['positive' if score > 0 else 'negative' if score < 0 else 'neutral' for score in sentiment_scores]

    # sentiment statistics per news category
    sentiment_df = pd.DataFrame([list(data['text']), list(data['full_name']), sentiment_scores, sentiment_category]).T
    sentiment_df.columns = ['text', 'author','score', 'sentiment_category']
    sentiment_df['score'] = sentiment_df.score.astype('float')
    sentiment_df.head()
    
    plot_sentiment_analysis(sentiment_df)
    
    return sentiment_df.head()
    
# Code Snippet for Plotting Results of Sentiment Analysis per Author   
def plot_sentiment_analysis(sentiment_df):     
    
    g = sns.FacetGrid(sentiment_df, col='author', height=10, col_wrap=3)

    g.map_dataframe(sns.countplot, data=sentiment_df, x='sentiment_category', palette="Greens_r")

    g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 18)
    g.set_xticklabels(rotation=0, size=16) 
    g.fig.subplots_adjust(top=0.5)
    g.fig.suptitle('Sentiment Analysis per Author', fontsize=20, weight='bold')

    axes = g.axes.flatten()
    axes[0].set_ylabel('Count')

    g.fig.tight_layout()
    
# Co-occurrence Word Pattern
# Utility Functions
# Returns the whole sentence, with preprocessed text
def preprocess_sentence(df): 
    word_list = []
    #remove punctuations
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', df) 
    #separate into words
    tokenized_word = word_tokenize(text)
    for word in tokenized_word:
        #filter stop-words
        if word not in all_stopwords:
            #stemming
            word = stem.stem(word) 
            #append to general list
            word_list.append(word) 
    return ' '.join(word_list) #rejoins the sentence without the stopwords

# Word Co-occurrence Analysis
def cooccurrence_analysis(data): 
    #extract sample from Dataset, Adding the whole dataset does not change the pattern
    corpus = data['text'].sample(frac=0.6, random_state=1) 
    #the CountVectorizer needs the inputs as list
    corpus = list(corpus)
    #extract the top most used words, uses my function to preprocess
    count_model = CountVectorizer(preprocessor=preprocess_sentence, max_features=20)
    #fits and transforms to my corpus    
    count_vect_data = count_model.fit_transform(corpus)
    #this is co-occurrence matrix in sparse csr format
    count_vect_datac = (count_vect_data.T * count_vect_data) 
    #fill same word co-occurence as zero, they have much higher numbers than the remaining words
    count_vect_datac.setdiag(0) 

    #create Dataframe
    count_vect_df = pd.DataFrame(count_vect_datac.A, columns=count_model.get_feature_names(), 
                                 index=count_model.get_feature_names())
    # creates a sum column to be used to order entries
    hour_count = count_vect_df.sum(axis=1)
    #sorter of rows and columns
    sorter = hour_count.sort_values(ascending=False).index 
    #sorts columns according to most occurred word
    count_vect_df = count_vect_df[sorter] 
    #sorts rows in the same order as columns
    count_vect_df = count_vect_df.reindex(sorter) 
    
    plot_cooccurrence_analysis(count_vect_df)

# Code Snippet for Plotting Heat Map 
def plot_cooccurrence_analysis(count_vect_df):
    mask = np.zeros_like(count_vect_df)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(14,10))
        ax = sns.heatmap(count_vect_df, mask=mask, vmax=70, vmin=20, linewidths=0.8, 
                         annot=False, cmap='Greens', annot_kws={"size": 10},cbar=True)
    plt.yticks(rotation=0) 
    plt.title('Words  Co-occurrence Pattern Plot', fontsize=16, weight='bold');    
