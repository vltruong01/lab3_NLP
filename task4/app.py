import torch
import torch.nn as nn
from flask import Flask, request, render_template
from transformers import BertTokenizer

# Define model parameters
input_size = 768
hidden_size = 512
output_size = 768
num_layers = 2

# Define the model architecture
class GeneralModelClass(nn.Module):
    def __init__(self):
        super(GeneralModelClass, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, src, trg):
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_outputs, _ = self.decoder(trg, (hidden, cell))
        attn_weights = self.attention(decoder_outputs)
        output = self.fc(attn_weights)
        return output

# Load the model
model = GeneralModelClass()
model.load_state_dict(torch.load('general_attention_model.pth'))
model.eval()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

app = Flask(__name__)

def translate_sentence(sentence):
    # Preprocess the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    src = inputs["input_ids"]
    trg = torch.zeros_like(src)  # Dummy target tensor
    
    # Pass the tensor through the model to get the translation
    with torch.no_grad():
        output = model(src, trg)
    
    # Convert the output tensor to a translated sentence
    translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_sentence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    input_sentence = request.form['input_sentence']
    translated_sentence = translate_sentence(input_sentence)
    return render_template('index.html', input_sentence=input_sentence, translated_sentence=translated_sentence)

if __name__ == '__main__':
    app.run(debug=True)