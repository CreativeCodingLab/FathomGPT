# From: https://github.com/openai/openai-cookbook/blob/8ab41ac37080d57277fae24089489beb417b90e9/examples/Get_embeddings_from_dataset.ipynb

# imports
import pandas as pd
import tiktoken

from openai.embeddings_utils import get_embedding
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# load & inspect dataset
input_datapath = "data/concepts.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath)
df = df.dropna()
print(df.head(2))

encoding = tiktoken.get_encoding(embedding_encoding)

df["embedding"] = df.concepts.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("data/concepts_embeddings.csv")
