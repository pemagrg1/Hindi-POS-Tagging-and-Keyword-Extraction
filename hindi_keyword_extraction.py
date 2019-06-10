from nltk.tag import tnt
from nltk.corpus import indian
import nltk
from nltk.tree import Tree

def hindi_model():
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger


def get_keywords(pos):
    grammar = r"""NP:{<NN.*>}"""
    chunkParser = nltk.RegexpParser(grammar)
    chunked = chunkParser.parse(pos)
    continuous_chunk = set()
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.add(named_entity)
                current_chunk = []
            else:
                continue
    return (continuous_chunk)

text = "इराक के विदेश मंत्री ने अमरीका के उस प्रस्ताव का मजाक उड़ाया है , जिसमें अमरीका ने संयुक्त राष्ट्र के प्रतिबंधों को इराकी नागरिकों के लिए कम हानिकारक बनाने के लिए कहा है ।"

model = hindi_model()
new_tagged = (model.tag(nltk.word_tokenize(text)))
print(new_tagged)
print()
print("====KEYWORDS===")
print(get_keywords(new_tagged))

#OUTPUT {'अमरीका', 'इराक', 'विदेश मंत्री', 'प्रस्ताव', 'नागरिकों', 'प्रतिबंधों', 'अमरीका संयुक्त राष्ट्र'}
