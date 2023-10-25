# train_hmm_model.py
import numpy as np
from hmmlearn import hmm
from chat.models import ChatMessage  # Import your Conversation model
import joblib  # Import the joblib library

# Fetch conversation data from the database
conversations = ChatMessage.objects.all()

# Prepare the sequences for training
sequences = [list(map(ord, conv.message)) for conv in conversations]
X = [item for sublist in sequences for item in sublist]

# Create and train the HMM model
model = hmm.MultinomialHMM(n_components=2)  # You can adjust the number of states
model.fit(np.array(X).reshape(-1, 1))

# Serialize and save the trained model to a file
joblib.dump(model, 'hmm_model.pkl')