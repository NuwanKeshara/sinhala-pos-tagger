import ast 
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

class tagger:

    def __init__(self):
        self.tag_dict = {0:'Other', 1:'Other' , 2:'NNC' , 3:'NNP' , 4:'PRP' , 5:'QBE' , 6:'QUE' , 7:'NDT' }
        self.MAX_SEQ_LENGTH = 300  # sequences greater than 100 in length will be truncated


        with open('word2id.txt') as file:
            data = file.read()
            self.d = ast.literal_eval(data)


        self.rnn_model = load_model('noun_rnn_model/')


    def encode_input(self, ilist):
        temp = []
        for word in ilist:
            temp.append(self.d[word])
        return temp


    def word_dict(self, sentence):
        word_dict = {}
        encoded = self.encode_input(sentence)
        for i in range(len(sentence)):
            try:
                word_dict[sentence[i]] = encoded[i]
            except:
                word_dict[sentence[i]] = ""
        # print(word_dict)
        return word_dict


    def int_to_tag(self, num):
        return self.tag_dict[num]

    def get_keys_from_value(self, d, val):
        return [k for k, v in d.items() if v == val]


    def result(self, input):
        input_encoded = self.encode_input(input)
        input_padded = pad_sequences([input_encoded], maxlen=self.MAX_SEQ_LENGTH, padding="pre", truncating="post")
        prediction = self.rnn_model.predict(input_padded)
        word = self.word_dict(input)
        dict = {}
        print(word)
        for i, pred in enumerate(prediction[0]):
            try:             
                if input_padded[0][i] != 0:
                    # print(self.get_keys_from_value(word, input_padded[0][i])," : " , np.argmax(pred))
                    # print()
                    dict[str(self.get_keys_from_value(word, input_padded[0][i]))] = str(self.int_to_tag(np.argmax(pred)))
            except:
                pass
        print(dict)
        return dict