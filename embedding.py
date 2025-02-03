import torch
from transformers import AutoTokenizer, AutoModel


class TextChunkerEmbedder:
    """
    Initializes the TextChunkerEmbedder class, loading the model and tokenizer.

    Args:
        model_name (str): The name of the pre-trained model (default: 'sentence-transformers/all-MiniLM-L6-v2').
        max_tokens (int):  Maximum tokens per chunk.
        stride (int): Token overlap between chunks.
    """

    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=200
    ):
        self.embeddings_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embeddings_model = AutoModel.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.stride = max_tokens // 2

    def map_document_embeddings(self, documents):
        """
        Processes a dictionary of documents by chunking their text and generating embeddings.

        This function takes a dictionary where keys are document names and values are the corresponding text.
        Each document's text is split into chunks using `chunk_text_by_tokens`, and an embedding is computed for each chunk
        using `embed_text`. The result is a dictionary mapping document names to a list of (chunk, embedding) tuples.

        Args:
            documents (dict[str, str]): A dictionary where keys are document names and values are their corresponding text content.

        Returns:
            dict[str, list[tuple[str, list[float]]]]: A dictionary mapping document names to a list of tuples.
            Each tuple contains a text chunk (str) and its corresponding embedding (list of floats).
        """

        mapped_documents = {}

        for doc_name, doc_text in documents.items():
            chunks = self.chunk_text_by_tokens(doc_text)
            mapped_embeddings = []

            for chunk in chunks:
                embeddings = self.embed_text(chunk)
                mapped_embeddings.append((chunk, embeddings))

            mapped_documents[doc_name] = mapped_embeddings

        return mapped_documents

    def embed_text(self, text):
        inputs = self.embeddings_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            embeddings = (
                self.embeddings_model(**inputs)
                .last_hidden_state.mean(dim=1)
                .squeeze()
                .tolist()
            )
        return embeddings

    def chunk_text_by_tokens(self, text):
        """
        Split text into chunks with the set size(max input sizee of LLM).

        Args:
            text (str): Text from the document.

        Returns:
            decoded_chunks[]: List with chunked text, where each ellement is one chunk.
        """
        tokens = self.embeddings_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        # Extract input IDs and overflowing tokens
        chunks = tokens["input_ids"]

        # Decode all chunks
        decoded_chunks = self.embeddings_tokenizer.batch_decode(
            chunks, skip_special_tokens=True
        )

        return decoded_chunks
