"""Vector indexer."""
from .client import get_client


class Indexer:
    def __init__(self, collection: str):
        self.collection = collection
        self.client = get_client()

    def index(self, doc):
        raise NotImplementedError
