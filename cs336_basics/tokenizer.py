from typing import Self, Iterable, Iterator
import cs336_basics.BPE
import regex as re
import multiprocessing
import pickle
import json
from tests.common import gpt2_bytes_to_unicode

#regex based pre-tokenizer used by GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.ranks = {pair: i for i, pair in enumerate(merges)}

        self.inverse_vocab = {v:k for k,v in self.vocab.items()}

        
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.inverse_vocab:
                    insert_max = max(vocab.keys()) + 1
                    self.vocab[insert_max] = token_byte

    def _merge(self, pre_token: tuple[bytes, ...]) -> list[bytes]:
        
        while True:
            best_rank = float("inf")
            best_index = None

            for i in range(len(pre_token) -1 ):
                pair = (pre_token[i], pre_token[i+1])
                
                if pair in self.ranks:
                    rank = self.ranks.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_index = i

            if best_index is None:
                return list(pre_token)

            merged = pre_token[best_index] + pre_token[best_index + 1]
            pre_token = pre_token[:best_index] + (merged,) + pre_token[best_index + 2:]
            

    def _pre_tokenize(self, text: str) -> tuple[bytes]:
        pre_tokenized_text = tuple()
        
        pre_tokenized_text+= tuple(i.encode("utf-8") for i in re.compile(PAT).findall(text))
                
        return pre_tokenized_text

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None) -> Self:
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)
    
    def _encode_chunk(self, text: str) -> list[int]:
        token_ids = []

        tokens = self._pre_tokenize(text)
        for token in tokens:
                initial_pre_token = tuple(bytes([b]) for b in token)
                merged = self._merge(initial_pre_token)
                for byte in merged:
                    token_ids.append(self.inverse_vocab.get(byte))
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs

        Args:
            text (str): Input text to encode.
        
        Returns:
            list[int]: Sequence of token ID's.
        """

        token_ids = []
        if not self.special_tokens:
            return self._encode_chunk(text)
        else:
            escape = [re.escape(t) for t in self.special_tokens]
            pattern = f"(" + '|'.join(escape) + ")"
            splitted = re.split(pattern, text)

            for part in splitted:
                if part in self.special_tokens:
                    token_ids.append(self.inverse_vocab.get(part.encode("utf-8")))
                else:
                    token_ids.extend(self._encode_chunk(part))
        return token_ids
    
    def encode_iterable(self, iterable : Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable string. This is required for memory-efficent tokenization. 
        Since we do not want to load large files into memory.

        Args:
            iterable (Iterable[str]): Iterable string to encode.

        Returns:
            Iterator[int]: Generator that yields Token IDs
        
        """
        for line in iterable:
            yield from self.encode(line)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of Token ids into text.

        Args:
            ids (list[int]): Sequence of token ids

        Returns:
            str: Text
        """
        concatenate_bytes = b""

        for id in ids:
            concatenate_bytes += self.vocab[id]
        return concatenate_bytes.decode("utf-8",errors="replace")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        vocab_filepath="tests/fixtures/gpt2_vocab.json",
        merges_filepath="tests/fixtures/gpt2_merges.txt",
        special_tokens=None
    )
    

    with open("cs336_basics/test.txt") as f:
        corpus_contents = f.read()

    print(tokenizer.encode(corpus_contents))
    #print(inst._pre_tokenize("the cat ate <|endoftext|> dasdas"))
    #print(inst.encode("the cat ate <|endoftext|>"))
