from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

documents = [
    "Virat Kohli is an Indian cricketer and former captain of the Indian national team.",
    "Sachin Tendulkar is widely regarded as one of the greatest batsmen in the history of cricket.",
    "Rohit Sharma is known for his elegant batting style and has captained the Indian cricket team.",
    "MS Dhoni is a former captain of the Indian cricket team and is known for his calm demeanor and finishing skills.",
    "Kapil Dev led India to its first Cricket World Cup victory in 1983."
]

query = 'tell me about Kapil Dev'

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], document_embeddings)[0]
# print("Cosine Similarities:", similarities)

index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[0]

print(query)
print(documents[index])
print(f"Similarity Score: {score:.4f}")