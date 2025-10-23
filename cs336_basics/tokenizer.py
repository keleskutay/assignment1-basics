from typing import Self, Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None) -> Self:
        raise NotImplementedError
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs

        Args:
            text (str): Input text to encode.
        
        Returns:
            list[int]: Sequence of token ID's.
        """
        raise NotImplementedError
    
    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable string. This is required for memory-efficent tokenization. 
        Since we do not want to load large files into memory.

        Args:
            iterable (Iterable[str]): Iterable string to encode.

        Returns:
            Iterator[int]: Generator that yields Token IDs
        
        """
        raise NotImplementedError
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of Token ids into text.

        Args:
            ids (list[int]): Sequence of token ids

        Returns:
            str: Text
        """
        raise NotImplementedError
