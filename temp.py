from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

# Save the model
with open('model.pkl', 'wb') as f:
  pickle.dump(model, f)