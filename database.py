import psycopg2
import psycopg2.extras
import os
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv


load_dotenv()


class DatabaseConnection:
    """
    Handls databse connection, embedding insertion and search.

    """

    _pool = None  # Static pool shared across all instances

    def __init__(self, min_conn=1, max_conn=5):
        if not DatabaseConnection._pool:  # Initialize the pool only once
            DatabaseConnection._pool = SimpleConnectionPool(
                min_conn,
                max_conn,
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
            )

    def get_connection(self):
        """Get a connection from the pool."""
        return DatabaseConnection._pool.getconn()

    def release_connection(self, connection):
        """Release the connection back to the pool."""
        DatabaseConnection._pool.putconn(connection)

    def insert_document_embeddings(self, mapped_documents):
        """Efficiently inserts multiple document chunks and embeddings."""
        connection = self.get_connection()
        try:
            with connection.cursor() as cursor:
                # Insert all documents (skip if already exists)
                document_names = [(doc_name,) for doc_name in mapped_documents.keys()]
                psycopg2.extras.execute_values(
                    cursor,
                    "INSERT INTO documents (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                    document_names,
                )

                # Get all document IDs in one query
                cursor.execute(
                    "SELECT id, name FROM documents WHERE name IN %s",
                    (tuple(mapped_documents.keys()),),
                )
                doc_id_map = {name: doc_id for doc_id, name in cursor.fetchall()}

                # Insert all chunks (skip duplicates)
                chunk_data = [
                    (doc_id_map[doc_name], chunk, embedding)
                    for doc_name, chunks_embeddings in mapped_documents.items()
                    for chunk, embedding in chunks_embeddings
                ]
                psycopg2.extras.execute_values(
                    cursor,
                    "INSERT INTO document_embeddings (document_id, chunk, embedding) VALUES %s ON CONFLICT DO NOTHING",
                    chunk_data,
                )

            connection.commit()
        except Exception as e:
            print(f"Error: {e}")
            connection.rollback()
        finally:
            self.release_connection(connection)

    def get_similar_embeddings(self, query_embedding, limit=5):
        """Retrieves the most similar chunks based on embedding distance."""
        connection = self.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.name, de.chunk, de.embedding <-> %s::vector AS distance
                    FROM document_embeddings de
                    JOIN documents d ON d.id = de.document_id
                    ORDER BY distance
                    LIMIT %s;
                    """,
                    (str(query_embedding), limit),
                )
                return cursor.fetchall()
        except Exception as e:
            print(f"Error: {e}")
            return []
        finally:
            self.release_connection(connection)

    @classmethod
    def close_pool(cls):
        """Closes the connection pool."""
        if cls._pool:
            cls._pool.closeall()
