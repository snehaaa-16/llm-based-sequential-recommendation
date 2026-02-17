# llm-based-sequential-recommendation
Sequential recommendation aims to predict the next item a user will interact with based on historical interactions.

Traditional LLM-based recommenders rely on autoregressive decoding with beam search, which introduces significant inference latency.

This project implements a hierarchical LLM architecture that:

- Compresses verbose item metadata into dense embeddings
- Models user interaction sequences efficiently
- Eliminates autoregressive beam search decoding via a projection-based ranking head

The goal is to improve inference efficiency while maintaining strong recommendation performance.
