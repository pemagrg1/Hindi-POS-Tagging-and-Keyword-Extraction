Created Date: 28 Sept 2018

# Hindi-POS-Tagging-and-Keyword-Extraction

Part of speech plays a very major role in NLP task as it is important to know how a word is used in every sentences. POS tagging is used mostly for Keyword Extractions, phrase extractions, Named Entity Recognition, etc. Before going further on POS tagging, I am assuming that you all know about part of speech as we all have studied grammar during school. Didn't we? But anyways let me give a brief explanation on it!


There are eight main Parts of Speech: Nouns(naming word), Pronouns(replaces a noun), Adjectives(describing word), Verbs(action word), Adverbs(describes a verb), Prepositions(shows relationships), Conjunctions(joining word) and Interjections(Expressive word). Most of it are further divided into sub-parts. Noun is divided into Proper Nouns, Common Nouns, Concrete Nouns etc.


Reminds you of school days?? Okay now lets start with Hindi Part of Speech Tagging.


Hindi Part of Speech Tagging is something that people are still doing research on as we have various techniques and libraries available for English Text and rarely for Hindi Text. [1] Manish and Pushpak researched on Hindi POS using a simple HMM based POS tagger with  accuracy of 93.12%. while [2]Nisheeth Joshi, Hemant Darbari and Iti Mathur also researched on Hindi POS using Hidden Markov Model with frequency count of two tags seen together in the corpus divided by the frequency count of the previous tag seen independently in the corpus. [3] S Phani Kumar Gadde, Meher Vijay Yeleti used CRF based tagger and Brants TnT (Brants, 2000), a HMM based tagger for hindi POS Tag where they got an acccuracy of 94.21%.


## So today we'll be using TNT tagger to tag Hindi words!


Lets say we have a text to tag


`text = "इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।"`


Let's use the already tagged data which is given in nltk to train the data.


`from nltk.tag import tnt
from nltk.corpus import indian
train_data = indian.tagged_sents('hindi.pos')
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)`


Let's Tag the text now! 


`tagged_words = (tnt_pos_tagger.tag(nltk.word_tokenize(text)))
print(tagged_words)
[OUTPUT]:
 [('इराक', 'NNP'), ('के', 'PREP'), ('विदेश', 'NNC'), ('मंत्री', 'NN'), ('ने', 'PREP'), ('अमरीका', 'NNP'), ('के', 'PREP'), ('उस', 'PRP'), ('प्रस्ताव', 'NN'), ('का', 'PREP'), ('मजाक', 'NVB'), ('उड़ाया', 'VFM'), ('है', 'VAUX'), (',', 'PUNC'), ('जिसमें', 'PRP'), ('अमरीका', 'NNP'), ('ने', 'PREP'), ('संयुक्त', 'NNC'), ('राष्ट्र', 'NN'), ('के', 'PREP'), ('प्रतिबंधों', 'NN'), ('को', 'PREP'), ('इराकी', 'JJ'), ('नागरिकों', 'NN'), ('के', 'PREP'), ('लिए', 'PREP'), ('कम', 'INTF'), ('हानिकारक', 'JJ'), ('बनाने', 'VNN'), ('के', 'PREP'), ('लिए', 'PREP'), ('कहा', 'VFM'), ('है', 'VAUX'), ('।', 'PUNC')]`


easy wasn't it?


The main issue here is that the nltk data is not complete. So we tend to get the tag "Unk" most of the time while tagging the words ex: ('वाशिंग', 'Unk'), ('मशीन', 'Unk').


#### How to overcome that?


1. we can stem the word as purposed by [1]
2. we can use probability with freq for next word as purposed by[2]
3. we can try handling the compound words as purposed by [3]
4. we can add more tagged sentences to NLTK hindi.pos
5. we can also use Google translator to translate and get the tag

I have tried using Google Translator API to handle the "Unk" tags by translating and getting the tags and then appending it to the NLTK Indian Corpus which gave a pretty good result.

### Keyword Extraction
run hindi_keyword_extraction.py file

`input_text = "इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।"`

`#OUTPUT {'अमरीका', 'इराक', 'विदेश मंत्री', 'प्रस्ताव', 'नागरिकों', 'प्रतिबंधों', 'अमरीका संयुक्त राष्ट्र'}` 

**REF:**

[1] "Hindi POS Tagger Using Naive Stemming : Harnessing Morphological Information Without Extensive Linguistic Knowledge" by Manish and Pushpak https://www.cse.iitb.ac.in/~pb/papers/icon08-hindi-pos-tagger.pdf

[2] "HMM BASED POS TAGGER FOR HINDI" by Nisheeth Joshi, Hemant Darbari and Iti Mathur. https://airccj.org/CSCP/vol3/csit3639.pdf

[3] "Improving statistical POS tagging using linguistic features for Hindi and Telugu" by S Phani Kumar Gadde, Meher Vijay Yeleti. https://researchweb.iiit.ac.in/~mehervijay.yeleti/papers/icon08-pos.pdf
