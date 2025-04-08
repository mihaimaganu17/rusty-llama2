#[repr(C)]
pub struct Transformer {
    // The configuration we read from the file
    pub config: Config,
    // The weights of the model
    pub weights: Weights,
    // All the buffers needed for one forward pass through the entire model
    pub state: State,
}

#[repr(C)]
pub struct Config {
    // Tranformer dimension and the size occupied by the embedding for each token
    embedding_size: usize,
    // Dimension for the hidden layer of the FFN network which cumullates two projections
    hidden_dim: usize,
    // Total number of layers in the transformer
    layer_count: usize,
    // Number of attention/query heads in the transformer
    heads_count: usize,
    // Number of key/value heads. Can be less than query heads because of multiquery
    kv_heads_count: usize,
    // Vocabulary size
    vocab_size: usize,
    // Maximum sequence length
    seq_len: usize,
}

#[repr(C)]
pub struct Weights {
    // Token embedding table (vocab_size, embedding_size)
    token_embedding_table: *const f32,
    // Weights for RMS norms, each with (layer_count, embedding_size) size
    w_rms_att: *const f32,
    w_rms_ffn: *const f32,
    // Weights for attention
    // Queries: (layer_count, embedding_size, heads_count * head_size)
    w_queries: *const f32,
    // Keys: (layer_count, embedding_size, kv_heads_count * head_size)
    w_keys: *const f32,
    // Values: (layer_count, embedding_size, kv_heads_count * head_size)
    w_values: *const f32,
    // Attentions scores output weights: (layer_count, heads_count * head_size, embedding_size)
    w_att_out: *const f32,

    // Weights for FFN
    w_projection1: *const f32,
    w_projection2: *const f32,
    w_projection_activation: *const f32,

    // Final RMS norm, before logits
    w_rms_final: *const f32,
    // (optional) classifier weights for the logits, on the last layer
    w_cls: *const f32,
}

#[repr(C)]
pub struct State {
    // Activation at the current token (embedding_size,)
    token_emb: *const f32,
    // Same embedding but connected with the residual branch (embedding_size,)
    token_emb_res: *const f32,
    // Additional buffer for convenience
    temp_buffer: *const f32,
    // Buffers for the 2 projections of the FFN's hidden layer, both (hidden_dim,)
    hidden_buffer1: *const f32,
    hidden_buffer2: *const f32,
    // Buffers to hold the results for queries, keys, values, all of size (embedding_size,)
    queries: *const f32,
    keys: *const f32,
    values: *const f32,
    // Buffer for attention scores (heads_count, seq_len)
    att_scores: *const f32,
    // Output logits (embedding_size, vocab_size)
    logits: *const f32,
    // Key - Value cache
    kv_cache: KVCache,
}

#[repr(C)]
pub struct KVCache {
    // Both of size (layer_count, seq_len, embedding_size)
    pub keys: *const f32,
    pub values: *const f32,
}

