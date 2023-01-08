from flask import Flask, render_template, jsonify, request
from flask_cors import CORS, cross_origin
import requests
import app.retriever as retriever

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/fact_check', methods=['GET', 'POST'])
def fact_check():
    '''
    input_dict = { 
      'highlighted_text' : highlighted_text, 
      'context_size' : context_size
    }
    '''

    data = request.get_json()

    highlighted_text = data['highlighted_text']
    context_size = data['context_size']
    search_results, URL, extracted_text, extracted_paragraph, similarity_score, title = retriever.fact_check_top_result(highlighted_text, context_size)
    
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
  # url = "http://127.0.0.1:5000"
  # highlighted_text = "pyramids of giza It stands 147 meters (481 feet) tall and was the tallest man-made structure in the world for over 3,800 years."
  # context_size = 200
  res_list = []

  # input_dict = { 
  #   'highlighted_text' : highlighted_text, 
  #   'context_size' : context_size
  # }
  # res = requests.post(url + '/fact_check', json=input_dict)
  # res_list += [{
  #   'URL' : res.json()['URL'], 
  #   'extracted_text' : res.json()['extracted_text'], 
  #   'extracted_paragraph' : res.json()['extracted_paragraph'], 
  #   'similarity_score': str(res.json()['similarity_score']), 
  #   'title' : res.json()['title']
  # }]

  # input_dict['search_results'] = res.json()['search_results']
  # input_dict['search_index'] = 1
  # res = requests.post(url + '/fact_check_index', json=input_dict)
  # res_list += [res.json()]
  return res_list

# if __name__ == "__main__":
#   app.run(debug=True)