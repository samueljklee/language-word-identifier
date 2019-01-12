# Language Word Identifier

### Author: Samuel Lee @ samueljklee@gmail.com

The goal of this program is to use the knowledge of n-grams to implement a language identification tool.
Currently it supports 4 languages: English, Malay, Dutch, Swedish
The reason for these languages is because of familiarity for EN and MS, DE and SV has similar language structure
(phonemes, letters used etc), I did noticed there are some issues of underfitting if not enough training are given and
bigram is used.

The current specifications of this program is by using Trigram and Quadgram as a combination to get the highest match
between the trained models and the test set (string).

How to run:
- To train the language model:
`python language_word_identifier.py train -i training`
Training sets should be saved in the `training` folder with the format of `[lang]_`

- To make predictions:
`python language_word_identifier.py predict [-d]`
`-d` flag is to show debugging messages

Example output:
```
> python language_word_identifier.py predict
Language:['de']   Number of n-gram: 6976
Language:['ms']  Number of n-gram: 4785
Language:['sv']  Number of n-gram: 5791
Language:['en']  Number of n-gram: 5704
Predicting words (type DONE to quit):
What to predict? > Hello World
Predicting: Hello World [Guessed: en][Prediction score: 38.03]
What to predict? > Apa khabar
Predicting: Apa khabar  [Guessed: ms][Prediction score: 35.89]
What to predict? > Gute Nacht
Predicting: Gute Nacht  [Guessed: de][Prediction score: 35.06]
What to predict? > Vad heter du?
Predicting: Vad heter du?       [Guessed: sv][Prediction score: 48.62]
What to predict? > DONE
Goodbye.
```

Reference: [AppliedMachineLearning Blog](https://appliedmachinelearning.blog/2017/04/30/language-identification-from-texts-using-bi-gram-model-pythonnltk/)
