import time
import torch
import Bert_Embeddings as be
from transformers import pipeline
from flask import Flask, redirect, url_for, request, render_template, jsonify
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

qa_pipeline = pipeline(
    "question-answering",
    model="bert-large-uncased-whole-word-masking-finetuned-squad",
    tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad"
)

path = 'source_sample_tesla.txt'
corpus_embeddings, embedder, corpus, sentences, para = be.load_model(path)

def sim_sent(query):
    start = time.time()
    queries = [query]
    results = be.ComputeSim(corpus_embeddings, embedder, queries)
    text = []
    for idx in results:
        text.append(corpus[idx]) 

    context = """ """.join(text)

    input_ids = tokenizer.encode(query, context)
    while True:
        if len(input_ids) < 384:
            break
        else:
            t = text.pop()
            context = """ """.join(text)
            input_ids = tokenizer.encode(query, context)

    return qa_pipeline({'question': f"""{query}""", 'context': context})['answer']
    
@app.route('/welcome')
def disp():
    return render_template('first.html')

@app.route("/a2/<string:question>/<string:answer>", methods=['POST','GET'])
def ans(question, answer):
    return render_template('second.html', q=question, a=answer)

@app.route('/a1', methods = ['POST', 'GET'])
def bert_qa():
    if request.method == 'POST':
        query = f"""{request.form['question']}"""
        query_answer = sim_sent(query)
        return redirect(url_for('ans', question = query, answer = query_answer))
        

app.run(host='0.0.0.0')

