import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer

class Utility:

    @staticmethod
    def extract_fields(dataframe, fields):
        extracted_data = []
        for row in dataframe:
            for field in fields:
                extracted_data.append(row[field])
        return extracted_data
    
    @staticmethod
    def extract_field_attributes(dataframe, field, attributes,key):
        extracted_data = []
        for row in dataframe:
            if row[field] in attributes:
                extracted_data.append(row[key])
        return extracted_data
    
    @staticmethod
    def get_full_language_name(lang):
        languages = {'en':'English', 'ja':'Japanese', 'fr':'french','zh':'Chinese',
            'es': 'Spanish','de':'German','hi':'hindi','ru':'Russian',
            'ko':'korean','te':'telugu','cn':'Chinese','it':'italian',
            'nl':'dutch', 'ta':'tamil', 'sv': 'swedish', 'th':'thai',
            'da':'danish','xx': '','hu':'hungarian','cs':'czech','pt':'portuguese',
            'is': 'icelandic', 'tr': 'turkish', 'nb': 'Norwegian Bokmal',
            'af': 'Afrikaans', 'pl': 'polish', 'he': 'hebrew', 'ar':'arabic',
            'vi': 'Vietnamese', 'ky':'Kirghiz', 'id': 'Indonesian', 'ro': 'Romanian',
            'fa': 'persian', 'no': 'Norwegian', 'sl': 'Slovenian', 'ps': 'Pushto',
            'el': 'Greek'}
        return languages[lang]

    @staticmethod
    def combine_data(row, fields):
        combined_text = ""
        for field in fields:
            if type(row[field]) == type([]):
                combined_text+=' '.join(row[field])
                combined_text+=" "
            else:    
                combined_text+=row[field]+" "
        return combined_text[:-1]

    @staticmethod
    def process_text(text, stop_words, stemmer):
        text = text.replace("uncredited","")
        text = re.sub('[^a-z\s]','',text.lower())
        text = [stemmer.stem(w) for w in text.split() if w not in set(stop_words)]
        return text

    @staticmethod
    def process_word(text, stemmer):
        text = re.sub('[^a-z\s]','',text.lower())
        return stemmer.stem(text)

    @staticmethod
    def highlight_words(x,word_list,stemmer):
        words_x = x.split(" ")
        for i in range(len(words_x)):
            if Utility.process_word(words_x[i].lower(),stemmer) in word_list:
                words_x[i] = "<strong> "+words_x[i]+" </strong>"
        return " ".join(words_x)

    @staticmethod
    def process_test_classification(text, stop_words, lemmatizer):
        text = re.sub('[^a-z\s,]', '', text.lower())
        text = [lemmatizer.lemmatize(w) for w in text.split() if w not in set(stop_words)]
        return ' '.join(text)

    