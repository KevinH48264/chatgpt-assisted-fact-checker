from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
# import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import re
import psutil
import os
# nltk.download('punkt')

MEMORY_CAP = 512000000000
num_sentences_to_use = 50

# from sentence_transformers import SentenceTransformer, util

# all-MiniLM-L6-v2: speed-14200, size-80Mb, server-640M
# all-distilroberta-v1: speed-4000, size-290Mb, server-1279M
# paraphrase-albert-small-v2: speed-5000(slow), size-43Mb (smallest), server-580M
# paraphrase-MiniLM-L3-v2 -- fastest, just less accurate
# model = SentenceTransformer('all-MiniLM-L6-v2') # or all-mpnet-base-v2
# model = SentenceTransformer('paraphrase-albert-small-v2')
# Load the model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE USING", device)

with open('tokenizer.pkl', 'rb') as f:
  tokenizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
  model = pickle.load(f)
  model.eval()
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# ENVIRONMENT VARS
load_dotenv()
my_cse_id = os.getenv('MY_CSE_ID')
dev_key = os.getenv('DEV_KEY')

# HELPER FUNCTIONS
# 1. GOOGLE SEARCH TOP SEARCH RESULT
def google_search(search_term, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=dev_key)
        res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
        return res['items']

def retrieve_top_search_result(fact_check_text):
    results = google_search(fact_check_text, my_cse_id, num=3, lr="lang_en")
    return results

# 2. WEB SCRAPE
# Helper functions to extract all text
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_URL(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return " ".join(t.strip() for t in visible_texts)

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_with_model(sentences):
    global MEMORY_CAP

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    process = psutil.Process(os.getpid())
    print("computing, CURRENT MEMORY USAGE: ", process.memory_info().rss / 1000000)
    if process.memory_info().rss > MEMORY_CAP:
        process.kill()

    with torch.no_grad():
        model_output = model(**encoded_input)
        
    process = psutil.Process(os.getpid())
    print("done computing, CURRENT MEMORY USAGE: ", process.memory_info().rss / 1000000)
    if process.memory_info().rss > MEMORY_CAP:
        process.kill()

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("sending back")
    return sentence_embeddings

def match_website_text(fact_check_text, website_text):
    global num_sentences_to_use
    print("matching website text with num_sentences_to_use: ", num_sentences_to_use)

    # Match closest sentence in website to fact check
    # Two lists of sentences
    fact_check_text_sentence = [fact_check_text]
    # TODO: use the tokenizer instead of splitting, or use nltk bc nltk isn't that intensive
    pattern = r"([^.]*\.)"
    website_sentences = re.findall(pattern, website_text)
    # website_sentences = nltk.sent_tokenize(website_text) # prevents us from using partition right now
    split_sentences = [website_sentences[i:i + num_sentences_to_use] for i in range(0, len(website_sentences), num_sentences_to_use)]
    website_sentences = [''.join(split) for split in split_sentences]

    # # print(len(website_sentences))
    website_sentences += fact_check_text_sentence
    print("NUM COMBINED ENTENCES: ", len(website_sentences))

    #Compute embedding for both lists
    process = psutil.Process(os.getpid())
    # embeddings1 = encode_with_model(fact_check_text_sentence)
    embeddings2 = encode_with_model(website_sentences)
    print("finished embeddings1, CURRENT MEMORY USAGE: ", process.memory_info().rss / 1000000)

    #Compute cosine-similarities
    print("getting embedding scores for this about of embeddings: ", len(embeddings2))
    cosine_scores = cosine_similarity(embeddings2[-1, :].unsqueeze(0), embeddings2[:-1, :])

    #Output the pairs with their score
    most_similar_idx = np.argmax(cosine_scores[0])
    similarity_score = cosine_scores[0][most_similar_idx]
    target_text = website_sentences[most_similar_idx]
    # print("Most similar sentence: ", target_text, " \nWith a similarity score of ", similarity_score)

    return similarity_score, target_text

# Extract relevant paragraph webpage 
def extract_paragraph(target_text, all_text, OFFSET):
    partitions = all_text.partition(target_text)
    return "..." + partitions[0][-OFFSET:] + " <b> " + partitions[1] + " </b> " + partitions[2][:OFFSET] + "..."

# use this for bringing other results
def extract_given_search_index(fact_check_text, search_results, context_size, search_index):
    print("extract_given_search_index!!")
    URL = search_results[search_index].get('link')
    # URL = 'https://en.wikipedia.org/wiki/Egyptian_pyramids'

    title = search_results[search_index].get('title')

    # Web scrape top google search result
    website_text = text_from_URL(URL)

    # Match closest sentence in website to fact check
    similarity_score, target_text = match_website_text(fact_check_text, website_text)

    # Extract relevant paragraph webpage 
    extracted_paragraph = extract_paragraph(target_text, website_text, context_size)

    if similarity_score < 0.25:
        return URL, "", "We could not scan this website. Please use the website link or use the next search result.", "Unknown", title

    return URL, target_text, extracted_paragraph, np.round(similarity_score, 2), title

# MAIN CODE below, runtime: 20 sec
# function to return extracted paragraphs and generating search results list
def fact_check(fact_check_text, check_top_n, context_size=100):
    # Retrieve top google search result
    search_results = retrieve_top_search_result(fact_check_text)

    # Retrieve URL, paragraph, and similarity_score from check_top_n sources
    URL_list, extracted_paragraph_list, similarity_score_list = extract_from_top_URLS(fact_check_text, search_results, check_top_n, context_size)
    
    # return top result based on similarity_score
    most_similar_idx = np.argmax(similarity_score_list)

    return URL_list, extracted_paragraph_list, similarity_score_list, most_similar_idx

# function to return top search index paragraph and search results
def fact_check_top_result(fact_check_text, context_size=100):
    print("Starting to find a top fact check source")

    # Retrieve top google search result
    search_results = retrieve_top_search_result(fact_check_text)

    # Retrieve URL, paragraph, and similarity_score for top result
    for i in range(3):
        URL, extracted_text, extracted_paragraph, similarity_score, title = extract_given_search_index(fact_check_text, search_results, context_size, i)
        if extracted_text != "":
            break

    return search_results, URL, extracted_text, extracted_paragraph, similarity_score, title

# MAIN CODE
# highlighted_text = "The pyramids are considered one of the Seven Wonders of the Ancient World."
# check_top_n = 1
# context_size = 200
# print()
# print("Trying to fact check: ", highlighted_text)
# print()
# search_results, URL, extracted_text, extracted_paragraph, similarity_score, title = fact_check_top_result(highlighted_text, context_size)
# print("Here is the most similar matching sentence: ", extracted_paragraph)
# print()
# print("Similarity score (0-1): ", similarity_score)
# print("Top Google Search result: ", URL)
# print("Title: ", title)
# print()

# print()
# print("Trying to fact check with second google search result")
# URL, extracted_text, extracted_paragraph, similarity_score, title = extract_given_search_index(highlighted_text, search_results, context_size, 1)
# print()
# print("Here is the most similar matching sentence: ", extracted_paragraph)
# print()
# print("Similarity score (0-1): ", similarity_score)
# print("Top Google Search result: ", URL)
# print("Title: ", title)
# print()