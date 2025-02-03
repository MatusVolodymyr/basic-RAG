import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import psycopg2
from dotenv import load_dotenv
import os
import psycopg2


class RAGSystem:
    def __init__(self, db_handler, embedding_handler, model_name="google/flan-t5-base"):
        """
        Initializes the RAG system with a database handler and the T5 model.

        Args:
            db_handler (DatabaseConnection): The database connection handler.
            model_name (str): The name of the pre-trained T5 model.
        """
        self.db_handler = db_handler
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.embedding_handler = embedding_handler

    def retrieve_relevant_chunks(self, query, top_k=5):
        """
        Retrieves the top-k most relevant document chunks for a given query.

        Args:
            query (str): The user's query.
            top_k (int): The number of relevant chunks to retrieve.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing the chunk text
            and its similarity score.
        """
        query_embedding = self.embedding_handler.embed_text(query)

        rows = self.db_handler.get_similar_embeddings(query_embedding, 2)

        # Return the relevant chunks along with similarity score
        relevant_chunks = [(row[0], row[1]) for row in rows]
        return relevant_chunks

    def generate_answer(self, question, top_k=5):
        """
        Generates an answer to the user's question using relevant document chunks.

        Args:
            question (str): The user's question.
            top_k (int): The number of relevant chunks to retrieve.

        Returns:
            str: The generated answer.
        """
        # Step 1: Retrieve relevant chunks from the database
        relevant_chunks = self.retrieve_relevant_chunks(question, top_k)

        # Step 2: Combine the relevant chunks into a context
        context = " ".join([chunk for _, chunk in relevant_chunks])

        # Step 3: Prepare the input for the model
        input_text = f"""You will be provided with some retrieved context, as well as the users query.
        Your job is to understand the request, and answer based on the context. 
        context: {context}
        question: {question}"""
        print("Input question: " + input_text)
        input_tokens = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=True
        )

        # Step 4: Generate the answer using T5
        with torch.no_grad():
            output = self.model.generate(input_tokens["input_ids"], max_length=512)

        # Step 5: Decode and return the generated answer
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
