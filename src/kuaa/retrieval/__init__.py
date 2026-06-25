"""Retrieval primitives: tokenizer, corpus builder, BM25, fusion.

This package contains the data-layer pieces of M2 Hybrid Search. None of
these modules import FastAPI, Jinja, or any web-layer concern — they are
plain functions/classes that take inputs and return outputs.

Top-level re-exports are deliberately minimal; consumers reach into
submodules (`from kuaa.retrieval.bm25 import BM25Index`) so call
sites read self-documentingly.
"""

from kuaa.retrieval.tokenize import tokenize

__all__ = ["tokenize"]
