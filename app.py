from flask import Flask, render_template, request
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load the T5 model for sequence-to-sequence language modeling (summarization)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    if request.method == 'POST':
        input_text = request.form['input_text']
        summary = abstractive_summarization(input_text, max_length=1000)  # Set the desired maximum length
    return render_template('index.html', summary=summary)

def abstractive_summarization(text, max_length):
    # Tokenize the input text into input_ids and attention_mask
    inputs = tokenizer.encode_plus(text, return_tensors="tf", max_length=2048, truncation=True, padding="max_length")

    # Generate the summary using the language model with the updated max_length
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=100, num_beams=4, early_stopping=True)

    # Decode the summary_ids into text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

if __name__ == '__main__':
    app.run(debug=True)
