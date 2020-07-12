import numpy as np
import math

class TF_IDF:
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
    
    def calculate_count_dic(self,all_doc):
        count_dic = {}
        for doc in all_doc:
            for word in doc:
                if word in count_dic:
                    count_dic[word]+=1
                else:
                    count_dic[word]=1
        return count_dic
    
    def calculate_IDF_dic(self,count_dic, len_data):
        idf_dic = {}
        for word in count_dic:
            idf_dic[word] = math.log(len_data/count_dic[word])
        return idf_dic
    
    def calculate_TFIDF_dic(self,doc_tf_dic, idf_dic):
        doc_tfidf_dic={}
        for word in doc_tf_dic:
            try:
                doc_tfidf_dic[word] = doc_tf_dic[word] * idf_dic[word]
            except Exception as e:
                print("Error in TFIDF calculation:",e)
        return doc_tfidf_dic
    
    def calculate_TFIDF_vector(self,words_list,word_dic,count_dic, tfidf_dic):
        tfidf_vector = [0.0] * len(words_list)
        for word in tfidf_dic:
            tfidf_vector[word_dic[word]] = tfidf_dic[word]
        return tfidf_vector
    
    def calculate_cosine_similarity(self,tfidf_vector, tfidf_vector_list):
        tfidf_vector_list = np.array(tfidf_vector_list)
        # cosine_similarity = []
        tfidf_vector = np.array(tfidf_vector)
        multiplication = np.sum(tfidf_vector_list * tfidf_vector,axis=0)
        vec1 = np.sum(np.square(tfidf_vector))
        vec2 = np.sum(np.square(tfidf_vector_list),axis=0)
        den = vec2*vec1
        multiplication = multiplication.astype(float)
        multiplication.reshape(multiplication.shape[0])
        den.reshape(den.shape[0])
        print("Multiplication:",multiplication.shape)
        print("den:",den.shape)
        return multiplication/den

