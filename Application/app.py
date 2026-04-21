from flask import Flask, render_template, request
import spacy
from spacy import displacy
from transformers import BertTokenizer

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def analyze_text(text):
    # Tokenize input text with BERT tokenizer
    bert_tokens = tokenizer.tokenize(text)
    
    # Convert the list of tokens to a string
    bert_tokens_str = ' | '.join(bert_tokens)
    
    # Process input text with spaCy
    doc = nlp(text)
    entities_html = displacy.render(doc, style='ent', page=True)
    
    return entities_html, bert_tokens_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    entities_html, bert_tokens = analyze_text(text)
    return render_template('result.html', entities_html=entities_html, bert_tokens=bert_tokens, input_text=text)

@app.route('/exit')
def exit():
    return render_template('exit.html')

if __name__ == '__main__':
    app.run(debug=True)
