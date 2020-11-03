from sentence_transformers import SentenceTransformer, util
import torch
import scipy.spatial
from nltk.tokenize import sent_tokenize
path = '/home/jarvis/Desktop/Bert/bert/source.txt'

def load_model(path):
    #embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    #embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    #embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    embedder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    #embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    with open(path, encoding='utf-8') as file:
        data = file.read()
    
    with open(path, encoding='utf-8') as file1:
        data1 = file1.read()
    para = data.split("\n")
    para = [p for p in para if len(p) > 1]
    sentences = []
    for i, p in enumerate(para):
        doc = sent_tokenize(p)
        for s in doc:
            sentences.append((i, s))

    corpus = sent_tokenize(data)
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor = True)
    return corpus_embeddings, embedder, corpus, sentences, para


def ComputeSim(corpus_embeddings, embedder, queries):
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_results = torch.topk(cos_scores, k=10)

        #distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
        #results = zip(range(len(distances)), distances)
        #results = sorted(results, key=lambda x: x[1])
        
    return top_results[1]



#cm, emb, c, sent, p = load_model(path)
#ComputeSim(cm, emb, ['who is the CEO of tesla?'], c)
