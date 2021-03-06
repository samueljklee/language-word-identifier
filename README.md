# Language Word Identifier

### Author: Samuel Lee @ samueljklee@gmail.com

The goal of this program is to use the knowledge of n-grams to implement a language identification tool.
Currently it supports 5 languages: English, Malay, Dutch, Swedish, Spanish
The reason for these languages is because of familiarity for EN and MS, DE and SV has similar language structure
(phonemes, letters used etc), and ES because it's widely used in the USA. I did noticed there are some issues of underfitting if not enough training are given and
bigram is used.

The current specifications of this program is by using Trigram and Quadgram as a combination to get the highest match
between the trained models and the test set (string). Each language are trained with around 250000 words, resulting the
number of unique tri- and quad- grams of around 4500 - 7000.

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
Language:['es']	 Number of n-gram: 5641
Language:['de']	 Number of n-gram: 6976
Language:['ms']	 Number of n-gram: 4785
Language:['sv']	 Number of n-gram: 5791
Language:['en']	 Number of n-gram: 5704
Predicting words (type DONE to quit):
What to predict? > Hello World
Predicting: Hello World [Guessed: en, Score: 1.0][Possible: sv, Score: 0.49]
What to predict? > Apa khabar
Predicting: Apa khabar  [Guessed: ms, Score: 1.0][Possible: sv, Score: 0.19]
What to predict? > Gute Nacht
Predicting: Gute Nacht  [Guessed: de, Score: 1.0][Possible: en, Score: 0.50]
What to predict? > Vad heter du?
Predicting: Vad heter du?       [Guessed: sv, Score: 1.0][Possible: en, Score: 0.25]
What to predict? > Jag alskar dig
Predicting: Jag alskar dig	[Guessed: sv, Score: 1.0][Possible: ms, Score: 0.60]
What to predict? > DONE
What to predict? > Mucho gusto.
Predicting: Mucho gusto.	[Guessed: es, Score: 1.0][Possible: en, Score: 0.69]
What to predict? > Adiós
Predicting: Adiós	[Guessed: es, Score: 1.0][Possible: ms, Score: 0.31]
Goodbye.
```

Reference: [AppliedMachineLearning Blog](https://appliedmachinelearning.blog/2017/04/30/language-identification-from-texts-using-bi-gram-model-pythonnltk/)
