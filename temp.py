# from sentence_transformers import SentenceTransformer
import pickle

# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Save the model
# with open('model.pkl', 'wb') as f:
#   pickle.dump(model, f)

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

with open('tokenizer.pkl', 'rb') as f:
  tokenizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Save the model
# with open('tokenizer.pkl', 'wb') as f:
#   pickle.dump(model, f)
# with open('model.pkl', 'wb') as f:
#   pickle.dump(model, f)

# Tokenize sentences
def encode_with_model(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

print("Sentence embeddings:")
print(encode_with_model(sentences))