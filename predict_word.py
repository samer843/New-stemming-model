import keras
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

char_to_index = {' ':0, 'ا': 1, 'ب': 2, 'ت': 3, 'ث': 4, 'ج': 5, 'ح': 6, 'خ': 7, 'د': 8, 'ذ': 9, 'ر': 10, 'ز': 11, 'س': 12, 'ش': 13, 'ص': 14, 'ض': 15, 'ط': 16, 'ظ': 17, 'ع': 18, 'غ': 19, 'ف': 20, 'ق': 21, 'ك': 22, 'ل': 23, 'م': 24, 'ن': 25, 'ه': 26, 'و': 27, 'ي': 28}
index_to_char = {v: k for k, v in char_to_index.items()}

word2="الكتب"
for w in word2:
    print(w)
def clean_text(text):
    # Remove non-Arabic characters and numerical values
    text = re.sub(r'[^ء-ي\s]', '', text)
    print(text)
    new_text = text.replace('أ', 'ا')
    print(text)
    new_text= new_text.replace('اّ', 'ا')
    new_text=new_text.replace('إ', 'ا')
    new_text = new_text.replace('ؤ','و')
    new_text = new_text.replace('ئ','ي')
    new_text = new_text.replace('ى','ي')
    new_text = new_text.replace('ة','ه')
    # Remove 'ء' character
    new_text = new_text.replace('ء', '')
    new_text = new_text.replace('ّ','')
    new_text = new_text.replace(" ","")
    
    return new_text
    

def change_to_digit(word):
    print(word)
    word=clean_text(word)
    print(word)
    word2=[]
    for w in word:
        word2.append(char_to_index[w])

    return word2

predict=change_to_digit(word2)
print(predict)
for i in range(14-len(predict)):
    predict.append(0)
#predict_padded= pad_sequence(predict, maxlen=14, padding='post', truncating='post')
print(predict)
p=np.array([predict])

print("p is {}".format(p))

loaded_model = keras.saving.load_model("root_to_words.keras")

y=loaded_model.predict(p)

binary_predictions = (y > 0.5).astype(int)  # Shape: (1, 14, 1)

# Remove batch dimension to get shape (14, 1)
binary_predictions = binary_predictions[0]
binary_predictions=binary_predictions.reshape(-1)
print(binary_predictions)
final_prediction=np.multiply(binary_predictions,p)
print(final_prediction)
predicted_word = "".join(index_to_char[num] for num in final_prediction[0] if num!=0)
print(predicted_word)
