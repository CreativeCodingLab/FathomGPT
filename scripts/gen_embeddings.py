# From: https://github.com/openai/openai-cookbook/blob/8ab41ac37080d57277fae24089489beb417b90e9/examples/Get_embeddings_from_dataset.ipynb

# imports
import pandas as pd
import tiktoken

import os
os.environ["OPENAI_API_KEY"] = 

from openai.embeddings_utils import get_embedding
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# load & inspect dataset
input_datapath = "data/concepts_desc.csv" 
df = pd.read_csv(input_datapath)
df = df[["concepts", "names", "links", "description", "related"]]
df = df.dropna()
df["combined"] = (
    #df.concepts.str.strip()+': '+df.description.str.strip() #+" ("+df.related.str.strip()+")"
    #df.concepts.str.strip() +" ("+ df.names.str.strip() +"): "+ df.description.str.strip()
    df.concepts.str.strip() +" ("+ df.names.str.strip() +")"
)

print(df['combined'][:10])

encoding = tiktoken.get_encoding(embedding_encoding)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model, ))
#df.to_csv("data/concepts_desc_embeddings2.csv")
#df.to_csv("data/concepts_names_desc_embeddings2.csv")
df.to_csv("data/concepts_names_embeddings2.csv")
