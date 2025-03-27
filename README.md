# Rusty Llama
This is a Rust port of Karpathy's llama2.c repository.

# Components

## Checkpoint file
The checkpoint file has 2 parts:
1. The configuration
2. The tranformer weights

### Configuation
The configuration contains metadata about the tranformer. Each of the following fields are
represented as a 4 byte unsigned integer.
1. Transformer dimension or embedding size: It is the total number of query heads multiplied by the
head size. This is effectively the `d_model` size of representations used as input to the multi-head
attention, which is the same size as the embedding size.
2. FFN size: Dimension for the feed-forward layers which follows each normalized
Multi-Head Attention layer
3. Number of layers: Total number of the model's layers
4. Number of query heads: Transformer dimension divided by the count of query heads gives us the
dimension size for each head, also called depth.
5. Number of key/value heads: Can be different because of multiquery, where the keys and value are
shared across all of the different attention "heads".
6. Vocabulary size: Total number of unique tokens used, usually 256 (byte-level)
7. Sequence length: Max sequence length that could be outputed by the model.

### Transformer weights
The transformer hash the following weights
1. Token embedding table: (V: Vocabulary size, D: Transformer dimension)
2. RMSNorm Attention weights: (L: number of layers, D: Transformer / Embedding dimension)
3. RMSNorm FeedForward weights: (L: number of layers, D: Transformer / Embedding dimension)
// Weights for attention heads
4. Query weights: (L: number of layers, D: Embedding dimension, Q: query heads * H: head size)
5. Key weights: (L: number of layers, D: Embedding dimension, KV: key-value heads * H: head size)
6. Value weights: (L: number of layers, D: Embedding dimension, KV: key-value heads * H: head size)
// This gets multiplied with the concatenation result of all the attention heads
7. Out weights: (L: number of layers, Q: query heads * H: head size, D: Embedding dimension)
// Weights for feed-forward net
8. W1: (L: Layer, F: feed-forward dim, D: Embedding dimension)
9. W2: (L: Layer, D: Embedding_dimension, F: feed-forward dim)
10. W3: (L: Layer, F: feed-forward dim, D: Embedding dimension)
11. RMSNorm final output weights: (D: Transformer / Embedding dimension)
12. (Optional) classifier weights for the logits, on the last layer.

### Run state
Keeps the internal state of a run through one attention head:
1. x: activation at the current time stamp (D: Embedding dimension)
2. xrb: activation at the current time stamp for the residual branch(D: Embedding dimension)
3. hidden_buffer: Buffer for the hidden dimension in the ffn (F: ffn hidden dimension)
4. hidden_buffer2: Buffer for the hidden dimension in the ffn (F: ffn hidden dimension)
5. query: Buffer for the query (D: Embedding dimension)
6. key: Buffer for the key (D: Embedding dimension)
7. value: Buffer for the value (D: Embedding dimension)
8. attention buffer: Buffer for scores/attention values (H: number of heads, S: seq length)
9. logits: output logits (H: number of heads, S: seq length)
10. key cache: maintains the activations from previous run (L: layer, S: sequence length, D: Emb)
11. value cache: maintains the activations from previous run (L: layer, S: sequence length, D: Emb)

# Calculus and arithmetic tangent
## Einsum notation
A great explanation of `einsum` could be found [here](https://stackoverflow.com/questions/26089893/understanding-numpys-einsum)
Shortly noting here. If we have 2 tensors
```
A = [0, 1, 2]
B = [[1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3]]
```
A has a single dimension/axis, lets call it. i = 3
B has 2 dimensions/axis, lets call it. i = 3 (axis 0) and j = 4 (axis 4).

If we would like to multiple A and B along the i axis and sum their product along j axis
to achieve a final single i axis tensor, we could write
```
np.einsum('i,ij->i', A, B)
```
Which would give: [0, 22, 76]

# Attention in Torch
## Prepare environment
1. Create a new virtual environment
```
python3 -m venv .llama2
```
2. Activate the new virtual environment
```
source .llama2/bin/activate
```
3. Install `torch`
```
pip install torch numpy
```
