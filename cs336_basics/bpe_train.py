import os
from typing import BinaryIO
import regex as re
from collections import defaultdict, Counter

#regex based pre-tokenizer used by GPT2

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def merge(word_counts: Counter, greatest_pair: tuple[bytes,bytes], new_token: bytes) -> dict[tuple[bytes], int]:
    merged_word_counts = defaultdict(int)

    p1,p2 = greatest_pair[0],greatest_pair[1]
    for symbol,freqs in word_counts.items():
        new_symbol = tuple()
        index=0
        while index < len(symbol):
            if index + 1 != len(symbol) and symbol[index] == p1 and symbol[index + 1] == p2:
                new_symbol += tuple((new_token,))
                index+=2
            else:
                new_symbol += tuple((symbol[index],))
                index+=1

        merged_word_counts[new_symbol] = freqs

    return merged_word_counts

def find_greatest_pair(pair_freqs: dict[tuple[bytes,bytes],int]) -> tuple[bytes,bytes]:
    highest_frequency = max(pair_freqs.values())
    lex_great_pair = max(tuple(k for k,v in pair_freqs.items() if v==highest_frequency))
    return lex_great_pair

def find_pair_freqs(token_freqs: dict[tuple[bytes], int]):
    """ Find frequencies of byte pairs from the coarse-grained frequencies table"""
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for symbols,freqs in token_freqs.items():
        for pair in zip(symbols, symbols[1:]):
            pair_freqs[pair] += freqs
    return pair_freqs


def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> Counter:
    """Before running pre-tokenization chunk split on special tokens then apply regex-based pre-tokenizer used by GPT2
        final output e.g. {(b'l', b'o',b 'w', b'e', b'r'): 12, (b'h', b'i',b'g', b'h'): 3, ...}
    """
    token_freqs =  Counter()

    escape = [re.escape(t) for t in special_tokens]
    pattern = f"{'|'.join(escape)}"
    sub_chunks = re.split(pattern, chunk)

    for sub_chunk in sub_chunks:
        for match in re.compile(PAT).findall(sub_chunk):
            match_bytes: tuple = tuple(bytes([i]) for i in match.encode("utf-8"))
            token_freqs[match_bytes] +=1
    
    return token_freqs


def pre_tokenize_corpus(input_path: str | os.PathLike, special_tokens: list[str]):
    word_counts = Counter()

    """Split file into chunks and apply pre-tokenizer for chunks"""
    with open(input_path, "rb") as __file:
        boundaries = find_chunk_boundaries(__file, 1, b"<|endoftext|>")

        for start,end in zip(boundaries[:-1], boundaries[1:]):
            __file.seek(start)
            chunk = __file.read(end - start).decode("utf-8", errors="ignore")
            word_counts.update(pre_tokenize_chunk(chunk, special_tokens))
    return word_counts

def _initialize_vocab(special_tokens: list[str]) -> dict[bytes,int]:
    #Initial 256 possible bytes
    initial_bytes = [bytes([i]) for i in range(256)]

    #Encode special tokens
    encoded_special_tokens = [token.encode("utf-8") for token in special_tokens]

    vocab = dict(enumerate(encoded_special_tokens + initial_bytes, start=0))

    return dict((v,k) for k,v in vocab.items())


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = _initialize_vocab(special_tokens)
    merges = []

    word_counts = pre_tokenize_corpus(input_path, special_tokens)

    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        pair_freqs = find_pair_freqs(word_counts)

        if not pair_freqs:
            break

        greatest_pair = find_greatest_pair(pair_freqs)
        
        new_token = greatest_pair[0] + greatest_pair[1]

        word_counts = merge(word_counts, greatest_pair, new_token)

        merges.append(greatest_pair)

        if vocab.get(greatest_pair) == None:
            vocab[new_token] = len(vocab)


if __name__ == "__main__":
    train_bpe(
        input_path="tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )