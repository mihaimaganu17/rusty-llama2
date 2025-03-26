import torch

A = torch.tensor([0, 1, 2])
B = torch.tensor([[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6]])

C = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6],
                  [5, 6],
                  [5, 6],
                  [5, 6],
                  [5, 6],
                  [5, 6],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12]])


def dot_product_attention(q, K, V):
    """
    Dot-Product Attention on one query
    Args:
        q: a single query, a tensor with shape [k]
        K: multiple keys (m keys, each key of size k), a tensor with shape [m, k]
        V: multiple values (m values, each value of size v), a tensor with shape [m, v]
    Returns:
        y: a tensor with shape [v]
    """
    # Perform retrieval comparing queries against the keys
    logits = torch.einsum("k,mk->m", q, K)
    print(logits)
    # Activation function
    weights = torch.softmax(logits.float(), dim=0)
    print(weights)
    # Get/retrieve values mapped to the according to weights
    return torch.einsum("m, mv->v", weights, V.float())

print(dot_product_attention(A, B, C))
