from fastapi import APIRouter
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = APIRouter()

class TodoItem(BaseModel):
    description: str

# Load the trained model
model = tf.keras.models.load_model('./app/models/todo_priority_model.h5')

# Load the tokenizer used during training
# Replace this with the appropriate code to load your tokenizer
with open('./app/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define a function to preprocess the input data
def preprocess_data(data):
    global tokenizer
    if not tokenizer:
        # Load the tokenizer object from a file
        with open('./app/models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences([data])
    max_length = model.input_shape[1]
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Define the API endpoint to predict the priority of a TODO item
@app.post('/predict')
def predict(item: TodoItem):
    # Preprocess the input data
    data = preprocess_data(item.description)
    
    # Make the prediction
    prediction = model.predict(data)
    
    # Return the predicted priority as a JSON response
    return {'priority': int(round(prediction[0][0]))}

# Define a default welcome message
@app.get('/')
def welcome():
    return {'message': 'Welcome to the TODO list app!'}

# Define a default welcome message
@app.get('/test')
def welcome():
    return {'message': 'Welcome to the TODO list app! test'}
