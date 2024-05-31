import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("HateSpeechDataset.csv")
df = df.dropna()  # Ensure there are no NaNs in the data
df.head()

# Loading pretrained Bert Transformer Model/Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to tokenize and embed sentences using BERT
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Taking the embeddings of the [CLS] token (first token) as the sentence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# Apply the function to each row in the 'Content' column with a progress bar
tqdm.pandas()  # Initialize tqdm for pandas
df['BERT_Embeddings'] = df['Content'].progress_apply(lambda x: get_bert_embeddings(x, tokenizer, model))

# Display the updated dataframe with embeddings
df.to_csv("EmbeddedHateSpeechDataset.csv")
#%%
