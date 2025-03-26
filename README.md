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
1. Transformer dimension: Presumably the count of all parameters of the model
2. FFN size: Dimension for the feed-forward layers which follows each normalized
Multi-Head Attention layer
3. Number of query heads
4. Number of key/value heads: Can be different because of multiquery, where the keys and value are
shared across all of the different attention "heads".
5. Vocabulary size: Total number of unique tokens used, usually 256 (byte-level)
6. Sequence length: Max sequence length that could be outputed by the model.


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
