import hashlib
import math
import re


class LocalEmbeddingClient:
    def __init__(self, dimensions: int = 3072):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector

        for index, token in enumerate(tokens):
            self._add_feature(vector, token, weight=1.0)
            if index + 1 < len(tokens):
                self._add_feature(vector, f"{token}_{tokens[index + 1]}", weight=0.5)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % self.dimensions
        sign = -1.0 if digest[0] & 1 else 1.0
        vector[index] += weight * sign