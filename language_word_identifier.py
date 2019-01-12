# Author: Samuel Lee @ samueljklee@gmail.com
# 
# The goal of this program is to use the knowledge of n-grams to implement a language identification tool.
# Currently it supports 4 languages: English, Malay, Dutch, Swedish
# The reason for these languages are because of familiarity for EN and MS, DE and SV has similar language structure
# (phonemes, letters used etc), I did noticed there are some issues of underfitting if not enough training are given and
# bigram is used. 
#
# The current specifications of this program is by using Trigram and Quadgram as a combination to get the highest match
# between the trained models and the test set (string). 
#
# Reference: https://appliedmachinelearning.blog/2017/04/30/language-identification-from-texts-using-bi-gram-model-pythonnltk/
# 


from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder, BigramAssocMeasures
import re
import os
import numpy as np
import string
import argparse 

global DEBUG
DEBUG = False
MODELS_PATH = "models/"
TRAIN_PATH = "training/"
FREQ_FILTER = 15

# Remove anything that is not lowercase and uppercase letters
def pre_processing(line):
    line = re.sub(r'[0-9\'!"#$%&\'()*+,-./:;]','', line).lower()
    return line

# Filter words
def filter_words(training_path, words):
    with open(training_path, 'r') as fpr:
        for i, row in enumerate(fpr):
            row = pre_processing(row)
            words.append(row)
            words.append(' ')

# Training language models 
def train_language(language, training_path):
    words = []
    filter_words(training_path, words)
    seq = ''.join(words)

    # Bigram
    bigram_finder = BigramCollocationFinder.from_words(seq)
    bigram_finder.apply_freq_filter(FREQ_FILTER)
    bigram_model = bigram_finder.ngram_fd.items()

    # Trigram
    trigram_finder = TrigramCollocationFinder.from_words(seq)
    trigram_finder.apply_freq_filter(FREQ_FILTER)
    trigram_model = trigram_finder.ngram_fd.items()

    # Quad 
    quadgram_finder = QuadgramCollocationFinder.from_words(seq)
    quadgram_finder.apply_freq_filter(FREQ_FILTER)
    quadgram_model = quadgram_finder.ngram_fd.items()

    bigram_model = sorted(bigram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    trigram_model = sorted(trigram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    quadgram_model = sorted(quadgram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    
    final_model = trigram_model + quadgram_model

    np.save(MODELS_PATH+language+'.npy', final_model)
    print("Language model for {} stored at {}".format(language, MODELS_PATH+language+'.npy'))

# Process model to store the result
def analyze_model():
    all_models = os.listdir(MODELS_PATH)
    language_model = [] 

    for model_file in all_models:
        language_name = re.findall(r'^[a-zA-Z]+', model_file)
        language_data = []
        
        model = np.load(MODELS_PATH+model_file)
        print("Language:{}\t Number of n-gram: {} ".format(language_name, len(model)))

        language_model.append((model_file, model, len(model)))

    return language_model 


def predict(test_string, models):
    # clean string
    test_string = pre_processing(test_string)

    bi_test = BigramCollocationFinder.from_words(test_string)
    tri_test = TrigramCollocationFinder.from_words(test_string) 
    quad_test = QuadgramCollocationFinder.from_words(test_string) 
    final_test = list(tri_test.ngram_fd.items()) + list(quad_test.ngram_fd.items())
    
    model_name = []

    for model in models:
        model_name.append(model[0])

    freq_sum = np.zeros(len(models))
    for ngram, freq in final_test:
        exists = 0

        for i, lang_model in enumerate(models):
            lang = lang_model[0]
            for k, v in lang_model[1]:
                total_ngram = lang_model[2]
                if k == ngram:
                    if DEBUG: print("Found", k, v, lang, total_ngram)
                    # normalizing to prevent freq/total to be zero 
                    freq_sum[i] = freq_sum[i] + (freq*10000)/total_ngram
                    exist = 1
                    break

            if not exists:
                freq_sum[i] += 1

        max_val = freq_sum.max()
        index = freq_sum.argmax()

    _max = 0
    _name = ""
    if DEBUG: print(list(zip(model_name, freq_sum)))
    for m,f in list(zip(model_name, freq_sum)):
        if f>_max:
            _name, _max = m, f

    return _name.split('.')[0], _max

def parse_arguments():
    parser = argparse.ArgumentParser(description="Identify language of given string")
    sub_parsers = parser.add_subparsers(help="help for subcommands", dest="mode")

    # train language arguments
    train_parser = sub_parsers.add_parser('train', help='train commands')
    train_parser.add_argument("-i", "--input", help="Training directory", required=True)

    # predict language arugments
    predict_parser = sub_parsers.add_parser('predict', help='predict commands')
    predict_parser.add_argument("-d", help="Debug messages on", action='store_true')

    return parser.parse_args()

def get_filepath(path):
    file_info = []
    if os.path.isdir(path):
        for file_path in os.listdir(path):
            name = re.findall(r'^[a-z]+', file_path).pop()
            file_info.append((name,TRAIN_PATH + file_path))

    return file_info

if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "train":
        pair_lang_train_path = get_filepath(args.input) 
        for lang, train_path in pair_lang_train_path:
            print("Training language {}".format(lang))
            train_language(lang, train_path)

    elif args.mode == "predict":
        if args.d:
            DEBUG=True
            print("Debug on")

        models = analyze_model()
        print("Predicting words (type DONE to quit):")
        while True:
            input_string = input("What to predict? ")
            if input_string == "DONE":
                break
            else: 
                prediction, score = predict(input_string, models) 
                score = re.findall(r'^[0-9]+.[0-9]{2}', str(score)) 
            print('Predicting: {}\t[Guessed: {}][Prediction score: {}]'.format(input_string, prediction, score[0]))
        
        print("Goodbye.")
   
