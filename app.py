import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load your Q&A CSV file
df = pd.read_csv("legal_qa.csv")

# Load a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert all questions to embeddings
question_embeddings = model.encode(df['question'].tolist())

# Create FAISS index and add vectors
index = faiss.IndexFlatL2(question_embeddings.shape[1])
index.add(question_embeddings)

# Save index and dataframe for future use
faiss.write_index(index, "legal_index.index")
df.to_pickle("legal_df.pkl")

# Function to retrieve the most relevant answer
def get_best_answer(user_input):
    query_embedding = model.encode([user_input])
    D, I = index.search(query_embedding, k=1)  # top 1 match
    # D is the distance (L2), I is the index
    best_distance = D[0][0]
    best_index = I[0][0]
    # Set a threshold for L2 distance (tune as needed)
    threshold = 200.0  # Try a higher value, adjust as needed
    print(f"[DEBUG] Closest distance: {best_distance}")
    if best_distance > threshold:
        return "Sorry, I don't know the answer to that."
    return df.iloc[best_index]['answer']

# Optional test loop (can be removed later)
if __name__ == "__main__":
    print("AI Legal Assistant is running. Type 'exit' to quit.")
    while True:
        question = input("\nAsk your question: ")
        if question.lower() == "exit":
            break
        answer = get_best_answer(question)
        print("Answer:", answer)
