# ABOUTME: Mints the stable per-sentence id shared by every extractor and every id-keyed artifact.
# ABOUTME: The sentence text is part of the digest so two sentences can never share an id.

import hashlib

_DIGEST_HEX = 8


def sentence_id(prefix: str, speaker: str, index: int, sentence: str) -> str:
    """Return ``{prefix}-{digest}-{index:03d}`` for one extracted sentence.

    ``prefix`` is the sitting date (``YYYY-MM-DD``) or, for the pre-1909
    archive, the year. ``index`` is the sentence's position in its speech.

    Hashing the sentence text is what keeps the id unique: a speaker's second
    speech on the same day restarts ``index`` at zero, and without the text
    every sentence of that speech would reuse the ids of the first.
    """
    source = f"{prefix}-{speaker}-{index}-{sentence}"
    digest = hashlib.md5(source.encode("utf-8")).hexdigest()[:_DIGEST_HEX]
    return f"{prefix}-{digest}-{index:03d}"


def split_id(sid: str) -> tuple[str, int]:
    """Recover ``(prefix, index)`` from any id in the current or legacy layout."""
    prefix, _digest, index = sid.rsplit("-", 2)
    return prefix, int(index)
