from typing import Self, Iterable, Iterator
import cs336_basics.BPE
import regex as re
import multiprocessing
import pickle

#regex based pre-tokenizer used by GPT2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    
    def _invert_vocab(self):
        return {v:k for k,v in self.vocab.items()}

    def _merge(self, pre_token: tuple[bytes, ...]) -> list[bytes]:
        
        ranks = {pair: i for i, pair in enumerate(self.merges)}

        while True:
            best_rank = float("inf")
            best_index = None

            for i in range(len(pre_token) -1 ):
                pair = (pre_token[i], pre_token[i+1])
                
                if pair in ranks:
                    rank = ranks[pair]
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_index = i

            if best_index is None:
                return list(pre_token)

            merged = pre_token[best_index] + pre_token[best_index + 1]
            pre_token = pre_token[:best_index] + (merged,) + pre_token[best_index + 2:]
            

    def _pre_tokenize(self, text: str) -> tuple[bytes]:
        pre_tokenized_text = tuple()

        #Split if special token exists
        escape = [re.escape(t) for t in self.special_tokens] if self.special_tokens else ""
        pattern = f"({'|'.join(escape)})"
        splitted = re.split(pattern, text)
        
        for txt in splitted:
            if self.special_tokens and txt in self.special_tokens:
                pre_tokenized_text+= tuple((txt.encode("utf-8"),))
            else:
                pre_tokenized_text+= tuple(i.encode("utf-8") for i in re.compile(PAT).findall(txt))

        return pre_tokenized_text

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None) -> Self:
        raise NotImplementedError
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs

        Args:
            text (str): Input text to encode.
        
        Returns:
            list[int]: Sequence of token ID's.
        """
        token_ids = []
        tokens = self._pre_tokenize(text)
        for pre_token in tokens:
            if self.special_tokens and pre_token.decode() in self.special_tokens:
                token_ids.append(self._invert_vocab()[pre_token])
                continue

            initial_pre_token = tuple(bytes([b]) for b in pre_token)
            merged = self._merge(initial_pre_token)
            for byte in merged:
                token_ids.append(self._invert_vocab()[byte])

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
    vocab,merges = cs336_basics.BPE.train_bpe(
        input_path="tests/fixtures/tinystories_sample.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    
    inst = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    ee = inst.encode("Hello, how are you?")
    print([inst.decode([x]) for x in ee])
    #print(inst._pre_tokenize("the cat ate <|endoftext|> dasdas"))
    #print(inst.encode("the cat ate <|endoftext|>"))
