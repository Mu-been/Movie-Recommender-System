# Movie-Recommender-System
Dataset [TMDB](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

## [Movie Search](https://datamining-282603.ue.r.appspot.com/search)

## Implementaion

Step 1: Calculate TF of each doc
```
def calculate_TF_dic(self,doc):
  doc_TF_dic =  {}
  for word in doc:
      if word in doc_TF_dic:
          doc_TF_dic[word] +=1
      else:
          doc_TF_dic[word] = 1
  for word in doc_TF_dic:
      doc_TF_dic[word] = doc_TF_dic[word]/len(doc)
  return doc_TF_dic
```

Step 2: Calculate counts of each term in all docs
```
def calculate_count_dic(self,all_doc):
  count_dic = {}
  for doc in all_doc:
      for word in doc:
          if word in count_dic:
              count_dic[word]+=1
          else:
              count_dic[word]=1
  return count_dic
```

Step 3: Calculate IDF values of each term

```
def calculate_IDF_dic(self,count_dic, len_data):
  idf_dic = {}
  for word in count_dic:
      idf_dic[word] = math.log(len_data/count_dic[word])
  return idf_dic
```

Step 4:  Calculate TF-IDF of each term in each doc
```
def calculate_TFIDF_dic(self,doc_tf_dic, idf_dic):
  doc_tfidf_dic={}
  for word in doc_tf_dic:
      try:
          doc_tfidf_dic[word] = doc_tf_dic[word] * idf_dic[word]
      except Exception as e:
          print("Error in TFIDF calculation:",e)
  return doc_tfidf_dic
```

Step 5: Convert TF-IDF value to vector rep
```
def calculate_TFIDF_vector(self,words_list,word_dic,count_dic, tfidf_dic):
  tfidf_vector = [0.0] * len(words_list)
  for word in tfidf_dic:
      tfidf_vector[word_dic[word]] = tfidf_dic[word]
  return tfidf_vector
```
Step 6: Calculate cosine similarity between two vectors
```
def calculate_cosine_similarity(self,tfidf_vector, tfidf_vector_list):
  tfidf_vector_list = np.array(tfidf_vector_list)
  tfidf_vector = np.array(tfidf_vector)
  multiplication = np.sum(tfidf_vector_list * tfidf_vector,axis=0)
  vec1 = np.sum(np.square(tfidf_vector))
  vec2 = np.sum(np.square(tfidf_vector_list),axis=0)
  den = vec2*vec1
  multiplication = multiplication.astype(float)
  multiplication.reshape(multiplication.shape[0])
  den.reshape(den.shape[0])
  return multiplication/den
```

## [Movie Classify](https://datamining-282603.ue.r.appspot.com/classify)

## Implementation: 
```
def initialize(self,x_train, y_train, genres):
  self.genres = genres
  self.len_unique_term = self.count_unique_term(x_train)
  self.len_train = x_train.shape[0]
  for genre in genres:
    self.class_prob[genre] = self.count_total_sample_class(x_train,y_train, genres.index(genre))/self.len_train
    self.total_word[genre] = self.count_total_word_class(x_train, y_train, genres.index(genre))
    self.word_count[genre] = self.count_word_occurance_class(x_train, y_train, genres.index(genre))
```

Calculate probability of test case for each genre
```
def predict(self,x_test):
  score = {}
  scores_detail = []
  for genre in self.genres:
    score[genre] = math.log(self.class_prob[genre])
    scores = {}
    scores['genre'] = genre
    scores['prob'] = score[genre]
    score_term = {}
    for term in x_test:
      if term in counter:
        score_term[term] = math.log(self.word_count[genre][term]+1/self.total_word[genre])
        score[genre] += math.log(self.word_count[genre][term]+1/self.total_word[genre])
      else:
        score_term[term] = math.log(1/self.total_word[genre])
        score[genre] += math.log(1/self.total_word[genre])
    scores['terms']=score_term
    scores['final'] = score[genre]
    scores_detail.append(scores)
  return score, scores_detail
```

## [Movie Recommendation](https://datamining-282603.ue.r.appspot.com/recommendation)

## Implementation:
```
def calculate_cosine_similarity():
    cosine_sim_all = cosine_similarity(tfidf_vector,tfidf_vector)
    return cosine_sim_all
