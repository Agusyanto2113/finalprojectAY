import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Sample conversation data for training
#conversations = [
#    ("Hi!", "Hello!"),
#    ("How are you?", "I'm doing well. How about you?"),
#    ("What's your favorite color?", "I'm just a chatbot. I don't have a favorite color."),
#    ("Tell me a joke.", "Why don't scientists trust atoms? Because they make up everything."),
#    ("How's the weather today?", "I'm not sure about the weather. I'm just here to chat."),
#    ("Goodbye", "Goodbye!"),
#    ("Where is batam located?","indonesia,kepulauan riau"),
#    # Add more training data here...
#]

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'conversation.txt')

# Initialize an empty list to store the conversation pairs
conversations = []

# Replace 'your_dataset.txt' with the actual filename of your dataset
with open(file_path, 'r') as file:
    lines = file.readlines()
    
    # Iterate through the lines, grouping them into pairs
    for i in range(0, len(lines), 2):
        user_message = lines[i].strip()
        bot_response = lines[i + 1].strip()
        conversation_pair = (user_message, bot_response)
        conversations.append(conversation_pair)

# Print the formatted conversation list
for pair in conversations:
    print(pair)




# Create and fit the tokenizer on the conversation data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([pair[0] for pair in conversations] + [pair[1] for pair in conversations])

# Tokenize and pad the conversation data
input_sequences = tokenizer.texts_to_sequences([pair[0] for pair in conversations])
output_sequences = tokenizer.texts_to_sequences([pair[1] for pair in conversations])

max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

# One-hot encode the output sequences
output_vocab_size = len(tokenizer.word_index) + 1
one_hot_output_sequences = tf.keras.utils.to_categorical(output_sequences, num_classes=output_vocab_size)

# Define the Seq2Seq model for training
input_seq = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_sequence_length)(input_seq)
lstm_layer = LSTM(32, return_sequences=True)(embedding_layer)
output = Dense(output_vocab_size, activation='softmax')(lstm_layer)

model = Model(inputs=input_seq, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(input_sequences, one_hot_output_sequences, epochs=2500000, verbose=1)

# Define a function to generate responses
def generate_bot_response(user_message, model, tokenizer):
    user_sequence = tokenizer.texts_to_sequences([user_message])
    user_sequence = pad_sequences(user_sequence, maxlen=max_sequence_length, padding='post')

    bot_response_sequence = model.predict(user_sequence)
    bot_response_sequence = np.argmax(bot_response_sequence, axis=-1)

    bot_response = tokenizer.sequences_to_texts(bot_response_sequence)[0]
    return bot_response

# Example conversation with the chatbot
user_input = "lima hotel terbaik di nagoya"
bot_response = generate_bot_response(user_input, model, tokenizer)
print(f"User: {user_input}")
print(f"Bot: {bot_response}")


# Calculate max_sequence_length
#max_sequence_length = max(
#    max(len(tokenizer.texts_to_sequences(pair[0])[0]), len(tokenizer.texts_to_sequences(pair[1])[0]))
#    for pair in conversations
#)

#print("Max Sequence Length:", max_sequence_length)

# Save the trained model for later use
model.save('chatbot_model.h5')
