import psycopg2
import json
from dotenv import load_dotenv
import os

class DatabaseConnection:
    """
    Handls databse connection.
    
    """
    def __init__(self):
        load_dotenv()
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT')
        self.dbname = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.connection = None

    def connect(self):
        """Establishes a database connection."""
        if not self.connection:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            #print('Connection opened')

    def disconnect(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            #print('Connection closed')

    def _execute_query(self, query, params=None):
        """Executes queries and returns the results.(in need of a fix)"""
        try:
            self.connect()
            cursor = self.connection.cursor()

            # Check if the query is a SELECT statement
            if query.strip().lower().startswith('select'):
                cursor.execute(query, params[0] if params else [])
                result = cursor.fetchall()
            else:
                # For non-SELECT queries (e.g., INSERT), iterate over parameters
                for param_set in params:
                    cursor.execute(query, param_set)

                result = None  # No results for non-SELECT queries

            self.connection.commit()
            cursor.close()
            return result
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
            if self.connection:
                self.connection.rollback()
        finally:
            self.disconnect()
    
    
    def insert_document_embeddings(self, mapped_documents):
        """
        Inserts a document chunk and its embedding into the database.
        
        Args:
            document_name (str): The name of the document.
            chunk (str): The chunked text.
            embedding (list): The embedding for the chunk.
        """

        params = []
        for doc_name, chunks_embeddings in mapped_documents.items():
            for chunk, embedding in chunks_embeddings:
                params.append((doc_name, chunk, embedding))

        insert_query = """
        INSERT INTO document_embeddings (document_name, chunk, embedding)
        VALUES (%s, %s, %s);
        """

        self._execute_query(insert_query, params)


    def get_similar_embeddings(self, query_embedding, limit=5):
        """
        Retrieves the most similar embeddings to the provided query embedding.
        
        Args:
            query_embedding (list): The embedding to compare against.
            limit (int): The maximum number of similar embeddings to retrieve.
        
        Returns:
            list: List of similar embeddings with their respective chunks and similarity scores.
        """

        query = """
        SELECT document_name, chunk, embedding <-> %s::vector AS distance
        FROM document_embeddings
        ORDER BY distance
        LIMIT %s;
        """
        params = []
        params.append((str(query_embedding), limit))
        return self._execute_query(query, params)
        
        