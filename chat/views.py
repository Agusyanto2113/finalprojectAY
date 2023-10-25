from django.shortcuts import render
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from hmmlearn import hmm
from .models import ChatMessage
from django.http import JsonResponse
from tensorflow.keras.models import load_model
# Create your views here.


#current_directory = os.getcwd()
#file_path = os.path.join(current_directory, 'conversation.txt')

# Initialize an empty list to store the conversation pairs
conversations = [
("hi!","hello!"),
("goodbye","goodbye!"),
("batam berlokasi dimana","indonesia, kepulauan riau"),
("berapa jumlah kecamatan di batam","jumlah kecamatan di batam ada 12 kecamatan"),
("berapa jumlah hotel di belakang padang","jumlah hotel di belakang padang adalah 0"),
("hotel terbaik di nongsa","montigo resort nongsa"),
("hotel terbaik di batam center","harris hotel batam center"),
("hotel terbaik di nagoya","nagoya hill hotel"),
("mall terbesar di batam","nagoya hill shopping mall"),
("mall apa saja yang ada di nagoya","grand mall batam dan bcs mall"),
("rekomendasi di nagoya malam hari","pergi ke thamrin city walk"),
("food court terbaik baik di nagoya","nagoya foodcourt dan a2 foodcourt dan winsor foodcourt"),
("makan enak di batam","sei enam seafood dan pantai cafe"),
("tempat baru di batam","infinty beach club"),
("rekomendasi tempat wisata batam","Wisata di Pulau Ranoh,Seaforest Adventure,Taman Kelinci,Ocarina Batam Theme Park,Kepri Coral Batam,Pulau Mubut,Pulau Belakang Padang,Jembatan Balerang"),
("rekomendasi perjalanan wisata di batam","Jalan-Jalan ke Kepri Coral Batam,Pulau Ranoh,Belanja di mall-mall rekomendasi,suasana food court di malam Harmoni"),
("rekomendasi di nagoya malam hari","pergi ke Thamrin city walk"),]

# Replace 'your_dataset.txt' with the actual filename of your dataset
#with open(file_path, 'r') as file:
#    lines = file.readlines()
    
    # Iterate through the lines, grouping them into pairs
#    for i in range(0, len(lines), 2):
#        user_message = lines[i].strip()
#        bot_response = lines[i + 1].strip()
#        conversation_pair = (user_message, bot_response)
#        conversations.append(conversation_pair)

# Print the formatted conversation list
#for pair in conversations:
#    print(pair)

# Create and fit the tokenizer on the conversation data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair[0] for pair in conversations] + [pair[1] for pair in conversations])

# Calculate max_sequence_length
#max_sequence_length = max(
#    max(len(tokenizer.texts_to_sequences(pair[0])[0]), len(tokenizer.texts_to_sequences(pair[1])[0]))
#    for pair in conversations
#)
# Tokenize and pad the conversation data
input_sequences = tokenizer.texts_to_sequences([pair[0] for pair in conversations])
output_sequences = tokenizer.texts_to_sequences([pair[1] for pair in conversations])

max_sequence_length = max(len(seq) for seq in input_sequences)
#input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
#output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# One-hot encode the output sequences
#output_vocab_size = len(tokenizer.word_index) + 1
#one_hot_output_sequences = tf.keras.utils.to_categorical(output_sequences, num_classes=output_vocab_size)

# Define the Seq2Seq model for training
#input_seq = Input(shape=(max_sequence_length,))
#embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_sequence_length)(input_seq)
#lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)
#output = Dense(output_vocab_size, activation='softmax')(lstm_layer)

#model = Model(inputs=input_seq, outputs=output)
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# Train the model
#model.fit(input_sequences, one_hot_output_sequences, epochs=600, verbose=1)

# Define the path to the folder containing the model
model_folder = os.path.join(os.path.dirname(__file__), '')  # 'models' is the name of the folder

# Load the trained RNN model from the specific folder
model_path = os.path.join(model_folder, 'chatbot_model.h5')
model = load_model(model_path)

# Define a function to generate responses
def generate_bot_response(user_message, model, tokenizer):
    user_sequence = tokenizer.texts_to_sequences([user_message])
    user_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length, padding='post')

    bot_response_sequence = model.predict(user_sequence)
    bot_response_sequence = np.argmax(bot_response_sequence, axis=-1)

    bot_response = tokenizer.sequences_to_texts(bot_response_sequence)[0]
    return bot_response


def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get('user_message')

        print("User Message:", user_message)        
        
        # Tokenize and pad the user's message
        user_sequence = tokenizer.texts_to_sequences([user_message])
        user_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length, padding='post')

        #get_reponsehmm = chatbot(user_message)

        # Generate a bot response
        bot_response = generate_bot_response(user_message, model, tokenizer)
    return JsonResponse({'response':bot_response})

def chatbot(user_message):
    
        user_input = user_message

        # Load the trained HMM model (you should adapt the path to your serialized model)
        model = hmm.MultinomialHMM(n_components=2)
        model = joblib.load('hmm_model.pkl')

        # Convert user input into a sequence of observations
        test_sequence = np.array(list(map(ord, user_input))).reshape(-1, 1)

        # Determine the most likely intent using the HMM model
        intent = model.predict(test_sequence)[0]

        # Retrieve chatbot response from the database based on the intent
        chatbot_response = ChatMessage.objects.filter(id=intent).first().chatbot_response

        return chatbot_response