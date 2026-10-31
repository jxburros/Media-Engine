"""Build the CLIP tokenizer fixture for the Rust unit tests (tests/data/clip_tokenizer).

A synthetic CLIP-shaped vocabulary (256 byte symbols, their `</w>` forms, merges learned from a
small corpus, then the start/end tokens) in two configurations shaped like the SDXL bundle's
`tokenizer` (pad `<|endoftext|>`) and `tokenizer_2` (pad `!`), plus the token ids transformers'
slow CLIPTokenizer (no ftfy) produces for a set of strings. Run with the reference interpreter:
    python native/sdxl/tests/make_tokenizer_fixture.py
"""

import collections
import json
import sys
from pathlib import Path

from transformers import CLIPTokenizer
from transformers.models.clip.tokenization_clip import bytes_to_unicode

OUT = Path(__file__).resolve().parent / "data" / "clip_tokenizer"

CORPUS = """
a photo of an astronaut riding a horse on the moon, highly detailed portrait of a woman with
red hair in the style of an oil painting, test window light studio lighting cinematic shot of a
cat sitting on the table, the quick brown fox jumps over the lazy dog, a beautiful landscape with
mountains and rivers at sunset, digital art trending on artstation, sharp focus, 8k resolution,
photorealistic rendering of a futuristic city at night with neon lights and rain, don't you'll
we're it's they've i'm she'd watercolor sketch of a lighthouse by the sea
"""


def learn_merges(text, count):
    byte_encoder = bytes_to_unicode()
    words = collections.Counter(
        "".join(byte_encoder[b] for b in w.encode("utf-8")) for w in text.lower().split()
    )
    vocab = {tuple(w[:-1]) + (w[-1] + "</w>",): n for w, n in words.items()}
    merges = []
    for _ in range(count):
        pairs = collections.Counter()
        for word, n in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += n
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        merges.append(best)
        merged = {}
        for word, n in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            merged[tuple(out)] = n
        vocab = merged
    return merges


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = list(bytes_to_unicode().values())
    vocab = symbols + [s + "</w>" for s in symbols]
    merges = learn_merges(CORPUS, 300)
    vocab += ["".join(m) for m in merges]
    vocab += ["<|startoftext|>", "<|endoftext|>"]
    encoder = {}
    for token in vocab:
        encoder.setdefault(token, len(encoder))
    special = {"lstrip": False, "normalized": True, "rstrip": False, "single_word": False, "special": True}
    start, end = encoder["<|startoftext|>"], encoder["<|endoftext|>"]
    configs = {
        "tokenizer": {
            "added_tokens_decoder": {
                str(start): {"content": "<|startoftext|>", **special},
                str(end): {"content": "<|endoftext|>", **special},
            },
            "bos_token": "<|startoftext|>",
            "clean_up_tokenization_spaces": True,
            "do_lower_case": True,
            "eos_token": "<|endoftext|>",
            "errors": "replace",
            "model_max_length": 77,
            "pad_token": "<|endoftext|>",
            "tokenizer_class": "CLIPTokenizer",
            "unk_token": "<|endoftext|>",
        },
    }
    configs["tokenizer_2"] = {
        **configs["tokenizer"],
        "added_tokens_decoder": {
            "0": {"content": "!", **special, "normalized": False},
            **configs["tokenizer"]["added_tokens_decoder"],
        },
        "pad_token": "!",
    }
    for name, config in configs.items():
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        (folder / "vocab.json").write_text(json.dumps(encoder, ensure_ascii=False))
        (folder / "merges.txt").write_text(
            "#version: 0.2\n" + "\n".join(" ".join(m) for m in merges) + "\n", encoding="utf-8"
        )
        (folder / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    texts = [
        "a test portrait",
        "",
        "A photo of an Astronaut riding a horse on the moon!",
        "Hello, World!!! It's 2024 -- café naïve 東京 \U0001F600 (really?) #art $5.99 & more...",
        "  multiple   spaces\tand\nnewlines 　 ideographic nbsp  ",
        "don't you'll we're I'M she'D they've it's",
        "<|startoftext|>inside<|endoftext|> tokens <|ENDOFTEXT|>",
        "ΟΔΟΣ Σίσυφος İstanbul",
        "é combining Å angstrom ﬁ ligature zero​width soft­hyphen",
        "123abc456 ٣٤ Ⅶ x²",
        "window" + ", window" * 60,
        " ".join(["highly detailed portrait of a woman"] * 20),
    ]
    expected = []
    for text in texts:
        row = {"text": text}
        for name in configs:
            tok = CLIPTokenizer.from_pretrained(str(OUT / name))
            assert tok.fix_text is None, "ftfy must not be installed for the reference ids"
            row[name] = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True).input_ids
        expected.append(row)
    (OUT / "expected.json").write_text(
        "[\n" + ",\n".join(json.dumps(row, ensure_ascii=False) for row in expected) + "\n]\n", encoding="utf-8"
    )
    print(f"wrote {OUT} ({len(encoder)} tokens, {len(merges)} merges, {len(texts)} strings)", file=sys.stderr)


main()
