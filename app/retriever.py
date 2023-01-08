from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
from sentence_transformers import SentenceTransformer, util
# import nltk
import numpy as np
import os
from dotenv import load_dotenv
# nltk.download('punkt') # Memory: 11MB

# all-MiniLM-L6-v2: speed-14200, size-80Mb, server-640M
# all-distilroberta-v1: speed-4000, size-290Mb, server-1279M
# paraphrase-albert-small-v2: speed-5000(slow), size-43Mb (smallest), server-580M
# model = SentenceTransformer('all-MiniLM-L6-v2') # or all-mpnet-base-v2
model = SentenceTransformer('paraphrase-albert-small-v2')

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
    # for result in results:
    #     print(result.get('link'))

    # top_URL = results[1].get('link')
    # print("Top URL hit: ", top_URL)
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

def match_website_text(fact_check_text, website_text):
    # Match closest sentence in website to fact check
    # Two lists of sentences
    fact_check_text_sentence = [fact_check_text]
    # website_sentences = nltk.sent_tokenize(website_text)
    website_sentences = website_text.split('.') # TODO: switch to NLTK because splitting is not very effective
    # print("website_text", website_text)
    # print("website sentences: ", website_sentences)

    #Compute embedding for both lists
    embeddings1 = model.encode(fact_check_text_sentence, convert_to_tensor=True)
    embeddings2 = model.encode(website_sentences, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    #Output the pairs with their score
    most_similar_idx = np.argmax(cosine_scores[0].cpu())
    similarity_score = cosine_scores[0][most_similar_idx].cpu().numpy()
    target_text = website_sentences[most_similar_idx]
    # print("Most similar sentence: ", target_text, " \nWith a similarity score of ", similarity_score)

    return similarity_score, target_text

# Extract relevant paragraph webpage 
def extract_paragraph(target_text, all_text, OFFSET):
  partitions = all_text.partition(target_text)
  # TODO: add <b> tag to put between partitions[1]?
  return "..." + partitions[0][-OFFSET:] + " <b> " + partitions[1] + " </b> " + partitions[2][:OFFSET] + "..."

def extract_from_top_URLS(fact_check_text, search_results, num_urls_to_check, context_size):
    URL_list, extracted_paragraph_list, similarity_score_list = np.array([]), np.array([]), np.array([])
    for i in range(num_urls_to_check):
        URL = search_results[i].get('link')

        # Web scrape top google search result
        website_text = text_from_URL(URL)

        # Match closest sentence in website to fact check
        similarity_score, target_text = match_website_text(fact_check_text, website_text)

        # Extract relevant paragraph webpage 
        extracted_paragraph = extract_paragraph(target_text, website_text, context_size)

        URL_list = np.append(URL_list, [URL])
        extracted_paragraph_list = np.append(extracted_paragraph_list, [extracted_paragraph])
        similarity_score_list = np.append(similarity_score_list, [similarity_score])

    return URL_list, extracted_paragraph_list, similarity_score_list

# use this for bringing other results
def extract_given_search_index(fact_check_text, search_results, context_size, search_index):
    URL = search_results[search_index].get('link')
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
    print("Starting to find a top fact check source")

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
    URL, extracted_text, extracted_paragraph, similarity_score, title = extract_given_search_index(fact_check_text, search_results, context_size, 0)

    return search_results, URL, extracted_text, extracted_paragraph, similarity_score, title

# MAIN CODE
# highlighted_text = "pyramids of giza It stands 147 meters (481 feet) tall and was the tallest man-made structure in the world for over 3,800 years."
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