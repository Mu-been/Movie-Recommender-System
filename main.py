from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import re
from naive_bayes import naive_bayes
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from utility import Utility

import pickle
from ast import literal_eval
from tf_idf import TF_IDF

app = Flask(__name__)

nltk.download('stopwords')
english_stemmer = SnowballStemmer("english")
genres =  np.array(['Action','Adventure','Animation','Comedy' ,'Horror','Romance'])

classification_X_file = 'x_train.pkl'
classification_Y_file = 'y_train.pkl'

x_file = open(classification_X_file,'rb')
y_file = open(classification_Y_file,'rb')

X_train = pickle.load(x_file)
Y_train = pickle.load(y_file)

classifier = naive_bayes()
classifier.initialize(X_train, Y_train, list(genres))

MEAN_MULTIPLIER = 1.19

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')


def initialize():
    df_metadata = pd.read_csv('tmdb_5000_movies.csv')
    df_combined = df_metadata
    df_combined['overview'] = df_combined['overview'].fillna("")
    df_combined['tagline'] = df_combined['tagline'].fillna('')
    df_combined['all_text'] = df_combined['overview']
    df_combined['all_text'] = df_combined['all_text'].apply(Utility.process_text,args=(stop_words,english_stemmer,))
    return df_combined

def process_test_classification(text):
    text = re.sub('[^a-z\s,]', '', text.lower())
    text = [english_stemmer.stem(w) for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def predict(text):
    text = process_test_classification(text)
    pred, score_details = classifier.predict(text.split(' '))
    print(pred)
    lst = []
    for i in range(genres.shape[0]):
        lst.append(pred[genres[i]])
    lst = np.array(lst)
    ids = lst > lst.mean()
    print(lst, lst.mean())
    print(ids)
    lst = lst[ids]
    pred_genre = genres[ids]
    arg = np.argsort(lst)
    arg = np.flip(arg,0)    
    new_score = []
    for i in arg:
        new_score.append(score_details[i])
    print(new_score)
    return {x:y for x,y in zip(pred_genre[arg],lst[arg])},new_score


tf_idf = TF_IDF()
tf_train_list = []

data = initialize()
for doc in data['all_text']:
    tf_train_list.append(tf_idf.calculate_TF_dict(doc))

count_dict = tf_idf.calculate_count_dict(data['all_text'].tolist())
idf_dict = tf_idf.calculate_IDF_dict(count_dict, len(tf_train_list))
tfidf_vector = []
word_list = sorted(count_dict.keys())
word_dict = {word:i for i,word in enumerate(word_list)}
for tf in tf_train_list:
    tfidf_vector.append(tf_idf.calculate_TFIDF_vector(word_list,word_dict,count_dict,tf_idf.calculate_TFIDF_dict(tf,idf_dict)))


def calculate_cosine_similarity():
    print("len:",len(tfidf_vector))
    cosine_sim_all = cosine_similarity(tfidf_vector,tfidf_vector)
    return cosine_sim_all

cosine_all_similarity = calculate_cosine_similarity()
print(len(cosine_all_similarity))

@app.route('/search',methods=['POST','GET'])
def search():
    try:
        raw_query = request.form['query']
        query = Utility.process_text(raw_query,stop_words,english_stemmer)
        query_tf_dict = tf_idf.calculate_TF_dict(query)
        print(query_tf_dict)
        query_tfidf_vector = tf_idf.calculate_TFIDF_vector(word_list,word_dict,count_dict,tf_idf.calculate_TFIDF_dict(query_tf_dict,idf_dict))
        sim_score = cosine_similarity([query_tfidf_vector],tfidf_vector)
        print(len(sim_score),len(sim_score[0]))
        sim_score = sim_score[0]
        sorted_indexes = np.argsort(sim_score).tolist()
        top_results = sorted_indexes[-20:]
        print(top_results)
        top_results.reverse()
        movies_list = data[['original_title','overview']].iloc[top_results]
        print("Here")
        movies_list['overview'] = movies_list['overview'].apply(Utility.highlight_words,args=(query,english_stemmer,))
        movies_list['cosine_similarity'] = sim_score[top_results]
        word_idf = {word:idf_dict[word] for word in query if word in idf_dict}
        print(word_idf)
    except Exception as e:
        print("Error:",e)
        return render_template('search_v1.html')
    return render_template('search_result_v1.html',list = movies_list,word_dict = word_idf)

@app.route('/classify', methods=['POST','GET'])
def classify():
    try:
        query = request.form['query']
    except Exception as e:
        print("Error:",e)
        return render_template('classify.html')
    result, score_details = predict(query)
    return render_template('classify_result.html', text_area_value = query,result = result,detail_scores = score_details)

@app.route('/recommendation', methods=['GET'])
def recommendation():
    id = 0
    try:
        id = int(request.args.get('id'))
    except Exception as e:
        print("Error:",e)
        return render_template('recommendation.html',flag = False,movie_list=data['original_title'].tolist(),size = len(data['original_title'].tolist()))
    print("ID:",id)
    cosine_id = cosine_all_similarity[id]
    print(len(cosine_id))
    sorted_indexes = np.argsort(cosine_id).tolist()
    top_results = sorted_indexes[-20:]
    top_results.reverse()
    movies_list = data[['original_title','overview']].iloc[top_results[1:]]
    movies_list['cosine_similarity'] = cosine_id[top_results[1:]]
    return render_template('recommendation.html',flag=True, movie_list=data['original_title'].tolist(),list = movies_list,size = len(data['original_title'].tolist()))

@app.route('/')
def home():
    return render_template('search_v1.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
