from flask import Flask, Response, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import requests
import retriever as retriever

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
url = "http://0.0.0.0:8080/"
# highlighted_text = "The Pyramid of Khufu is the largest Egyptian pyramid. It is the only one of the Seven Wonders of the Ancient World still in existence, despite its being the oldest wonder by about 2,000 years."
highlighted_text = "What you do makes a difference, and you have to decide what kind of difference you want to make."

context_size = 100

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route('/fact_check', methods=['GET', 'POST'])
def fact_check():
    '''
    input_dict = { 
      'highlighted_text' : highlighted_text, 
      'context_size' : context_size
    }
    '''
    # WORKS WITH THIS COMMENTED OUT
    # print("fact checking!")
    current_URL = ""
    if request.method == 'GET':
      global highlighted_text
      global context_size
    if request.method == 'POST':
    #   print("WASSUP")

      data = request.get_json()

      highlighted_text = data['highlighted_text']
      context_size = data['context_size']
      if 'current_URL' in data.keys():
        current_URL = data['current_URL']

    # print("SEARCHING")
    print("search text: ", highlighted_text)
    search_results, URL, extracted_text, extracted_paragraph, similarity_score, title = retriever.fact_check_top_result(highlighted_text, context_size, current_URL)
    # search_results = retriever.fact_check_top_result(highlighted_text, context_size)

    # print("DONE SEARCHING")
    return jsonify({
        'search_results' : search_results,
        'URL' : URL, 
        'extracted_text' : extracted_text, 
        'extracted_paragraph' : extracted_paragraph, 
        'similarity_score': str(similarity_score), 
        'title' : title
    })

@app.route('/fact_check_index', methods=['GET', 'POST'])
def fact_check_index():
    '''
    input_dict = { 
      'highlighted_text' : highlighted_text, 
      'context_size' : context_size,
      'search_results' : 'search_results',
      'search_index' : search_index
    }
    '''

    data = request.get_json()

    search_results = data['search_results']
    search_index = data['search_index']
    highlighted_text = data['highlighted_text']
    context_size = data['context_size']

    URL, extracted_text, extracted_paragraph, similarity_score, title = retriever.extract_given_search_index(highlighted_text, search_results, context_size, search_index)
    
    return jsonify({
        'URL' : URL, 
        'extracted_text' : extracted_text, 
        'extracted_paragraph' : extracted_paragraph, 
        'similarity_score': str(similarity_score), 
        'title' : title
    })

@app.route('/')
def index():
  print("RUNNING APP!")
  res_list = []

  input_dict = { 
    'highlighted_text' : highlighted_text, 
    'context_size' : context_size
  }
  res = input_dict
  res = requests.post(url + '/fact_check', json=input_dict)
#   print(res.json())
  res_list += [{
    'URL' : res.json()['URL'], 
    'extracted_text' : res.json()['extracted_text'], 
    'extracted_paragraph' : res.json()['extracted_paragraph'], 
    'similarity_score': str(res.json()['similarity_score']), 
    'title' : res.json()['title']
  }]
  print("DONE")

  # input_dict['search_results'] = res.json()['search_results']
  # input_dict['search_index'] = 1
  # res = requests.post(url + '/fact_check_index', json=input_dict)
  # res_list += [res.json()]
  return jsonify({
    'URL' : res.json()['URL'], 
    'extracted_text' : res.json()['extracted_text'], 
    'extracted_paragraph' : res.json()['extracted_paragraph'], 
    'similarity_score': str(res.json()['similarity_score']), 
    'title' : res.json()['title']
  })

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=8080)