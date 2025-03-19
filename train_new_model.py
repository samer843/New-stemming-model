#required libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import numpy as np
import re
import time


# data preprocessing
def clean_text(text):
    
    #letter normalization
    new_text = text.replace('أ', 'ا')
    
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
    




    
# decimal numeric repersentation for each arabic letters
char_to_index = {' ':0, 'ا': 1, 'ب': 2, 'ت': 3, 'ث': 4, 'ج': 5, 'ح': 6, 'خ': 7, 'د': 8, 'ذ': 9, 'ر': 10, 'ز': 11, 'س': 12, 'ش': 13, 'ص': 14, 'ض': 15, 'ط': 16, 'ظ': 17, 'ع': 18, 'غ': 19, 'ف': 20, 'ق': 21, 'ك': 22, 'ل': 23, 'م': 24, 'ن': 25, 'ه': 26, 'و': 27, 'ي': 28}

# the maximum squence length acceptable
max_sequence_length = 14

embedding_dim = 29  #  embedding dimension
hidden_units = 29  #  LSTM units



num_epochs = 3  #  the number of epochs
batch_size = 128  #  batch size

#dataset generation stage
print("we entered the data generation stage")
start=time.time()
roots=[]
words=[]
end=time.time()
t=end-start

# for loop to generate 3 literal words-roots dataset so three loop used
for i in range(1,28,1):

    for f in range(1,28,1):

        for y in range(1,28,1):
            # open the pattern file and reaad each pattern the patterns file contain ltters "A", "B", and "C"
            # the letter "A" represent arabic letter "ف"
            # the letter "B" repersent arabic letter "ع"
            # the letter "C" repersent arabic letter "ل"
            with open("words.txt","r",encoding="utf-8")as file:
                lines=file.readlines()
                
                w=[]
                r=[]
                word=[]
                for l in lines:
                    # read each pattern and do the preprocessing
                    l=clean_text(l)
                    for char in l:
                        
        
                        if char=="A":
                            w.append(i)# replace the pattern "A" with the first loop "ِِA" her  represent the "ف" in arabic

                
                            r.append(1)# append 1 to the root as the letter among the root
                            
                        

                        elif char=="B":
                            w.append(f) # replace the pattern "B" with the second loop "B" her  represent the "ع" in arabic
                            r.append(1)  # append 1 to the root as the letter among the root


                     # replace the pattern "C" with the third loop "C" her  represent the "ل" in arabic
                        elif char=="C":
                            w.append(y)
                            r.append(1)
                        elif char=="\n":
                            pass
                        else:
                            r.append(0)
                            w.append(char_to_index[char]) # change the letter to their numerical repersentation
                    words.append(w)  #append the decimal repersentation of the word to the words data

                    w=[]
                
                    roots.append(r)# append the binary repersentation of the root to roots data
                    r=[]
                
end=time.time()

print("data generation tooks {} seconds".format(end-start))

print(words[0])

#data preparation stage
print("entering data preparation stage")



### padding stage. pad the words data and roots data with zeros to fill 14 vector for each sample
root_indices_padded = pad_sequences(roots, maxlen=max_sequence_length, padding='post', truncating='post')
words_indices_padded = pad_sequences(words, maxlen=max_sequence_length, padding='post', truncating='post') 


print("words indices padded {}".format(words_indices_padded[0]))
print("root indices padded {}".format(root_indices_padded[0]))




# training data
X=np.array(words_indices_padded)

Y=np.array(root_indices_padded)






# Use callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)



####
# model design

model = Sequential()
model.add(Embedding(input_dim=29, output_dim=embedding_dim))# embedding layer
model.add(Bidirectional(LSTM(units=hidden_units, return_sequences=True)))# first bidirectional lstm layer
model.add(Dropout(0.3))#dropout layer
model.add(Bidirectional(LSTM(units=hidden_units, return_sequences=True))) # 2nd bidrectional lstm alyer
model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # Output a sequence


#### Compile and Train the model
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


# Train the model
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, reduce_lr]
)




# Save the model
model.save('root_to_words.keras')

# Plot the training and validation loss and accuracy
def plot_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot as a file
    plt.savefig(save_path)

    # Show the plot
    plt.show()

# Call the plot function
plot_history(history)
