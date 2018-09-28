from nltk.tag import tnt
from nltk.corpus import indian
from googletrans import Translator
import nltk
import re

translator = Translator()
sentence_id = 0

text = "बाबा रामदेव के गुरुकुल का आज अमित शाह करेंगे उद्घाटन, योगगुरु बोले- बच्‍चों को वेद भी पढ़ाते हैं और योग भी कराते हैं"
model_path = "data/hindi.pos" #Copy hindi.pos from NLTK corpus

def train_hindi_model(model_path):
    train_data = indian.tagged_sents(model_path)
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger


def get_sentId(model_path):
    ids = re.compile('<Sentence\sid=\d+>')
    with open(model_path, "r+") as temp_f:
        content = temp_f.readlines()
        for i in content:
            id_found = (ids.findall(i))
            if id_found:
                id_found = str(id_found).replace("['<Sentence id=", "").replace(">']", "")
                id = int(id_found)
    id = id + 1
    return id


def tag_words(model,text):
    tagged = (model.tag(nltk.word_tokenize(text)))
    return tagged


def handle_UNK(tagged_words, model_path, sentence_id):
    with open(model_path, "r+") as f1:
        result_list = []
        for nep_word, tag in tagged_words:
            if tag == "Unk":
                x = translator.translate(nep_word)
                if x is not None:
                    str1 = str(x)
                    new_str = str1.split()
                    for j in new_str:
                        if re.search('^text=', j, re.I):
                            word = j.replace("text=", ",").replace(",", "")
                            word = str(word)
                            # pos=nltk.pos_tag(word)
                            pos = nltk.tag.pos_tag([word])
                            # print (i, pos)
                            for en_word, tag in pos:
                                result = nep_word + "_" + (tag) + " "
                                result_list.append(result)

            else:
                result = nep_word + "_" + (tag) + " "
                result_list.append(result)

        writing_word = str("\n<Sentence id=") + str(sentence_id) + ">\n"
        output = writing_word + "".join(result_list) + "\n</Sentence>\n</Corpora>"
        for line in f1.readlines():
            f1.write(line.replace("</Corpora>", ""))
        f1.write(output)


sentence_id = (get_sentId(model_path))
print (sentence_id)

model = train_hindi_model(model_path)
tagged_words = tag_words(model,text)

print ("=================================Tagged words=================================\n",tagged_words,"\n")

handle_UNK(tagged_words,model_path,sentence_id)

#retrain the model
model = train_hindi_model(model_path)
new_tagged_words =  tag_words(model,text)
print ("=================================New Tagged words=================================\n",new_tagged_words,"\n")

with open("result/handling_UNK_output.txt","a") as output_file:
    output_file.write("[INPUT]\n")
    output_file.write(text)
    output_file.write("\n[BEFORE RE TRAIN]\n")
    output_file.write(str(tagged_words))
    output_file.write("\n[AFTER RE TRAIN]\n")
    output_file.write(str(new_tagged_words))
