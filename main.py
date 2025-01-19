import database
import file_loader
import embedding
import generation_model

# Load text documents from the specified directory
documents = file_loader.load_texts_from_directory('documents')

# Initialize the handler for preprocessing the documents (chunking and embedding them)
embedding_handler = embedding.TextChunkerEmbedder('sentence-transformers/all-MiniLM-L6-v2', max_tokens=200)

# Chunk the documents and generate embeddings
document_embeddings = embedding_handler.map_document_embeddings(documents)

# Initialize the handler for interacting with the database
db_connection = database.DatabaseConnection()

# Insert document embeddings into the database (need a fix to ensure no duplicates are stored)
#db_connection.insert_document_embeddings(document_embeddings)

# Prompt the user for a query
user_query = input('What is your question?\n')

# Initialize the RAG (Retrieval-Augmented Generation) system with the database and embedding handler
rag_system = generation_model.RAGSystem(db_connection, embedding_handler)

# Generate and print the answer based on the user's query
print('Answer:' + rag_system.generate_answer(user_query))
