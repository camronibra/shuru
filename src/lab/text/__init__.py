"""Text processing for Persian-script text, in pure Python.

- ``normalise``: make equivalent spellings identical (yeh, kaf, zero-width characters).
- ``tokenise``: split text into words, numbers and punctuation, by script.
- ``dedup``: MinHash signatures for near-duplicate detection.
"""
