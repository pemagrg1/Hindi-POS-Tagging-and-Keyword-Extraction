from nltk.tag import tnt
from nltk.corpus import indian
import nltk

def hindi_model():
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger

text = "इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।"

model = hindi_model()
new_tagged = (model.tag(nltk.word_tokenize(text)))
print(new_tagged)

# with open("result/output.txt","a") as output_file:
#     output_file.write("\n[INPUT]\n")
#     output_file.write(text)
#     output_file.write("\n[OUTPUT]\n")
#     output_file.write(str(new_tagged))
