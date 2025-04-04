#[unsafe(no_mangle)]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust!");
}

pub unsafe fn softmax(input: *mut f32, size: usize) {
    unsafe {
        // Find the max value for numerical stability. This avoids the exponential function to
        // overflow in the case of large numbers and avoids nans because an exponential overflow of
        // a large number could give us inf/inf: https://jaykmody.com/blog/stable-softmax/
        let mut max = *input.add(0);
        for idx in 1..size {
            if *input.add(idx) > max {
                max = *input.add(idx);
            }
        }
        // Compute the sum of the exponentials of all the values in the array
        let mut sum = 0.0f32;
        for idx in 0..size {
            *input.add(idx) = (*input.add(idx) - max).exp();
            sum += *input.add(idx);
        }
        // Divide each element by the sum of exponentials
        for idx in 0..size {
            *input.add(idx) /= sum;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn multihead_attention(
    head_count: usize,
    kv_head_count: usize,
    head_size: usize,
    layer: usize,
    seq_len: usize,
    embedding_size: usize,
    current_position: usize,
    attention_scores: *mut f32,
    queries: *mut f32,
    keys_cache: *mut f32,
    values_cache: *mut f32,
    weigh_attention: *mut f32,
    output_weights: *mut f32,
    heads_activation: *mut f32, // (embedding_size,) == (head_count * head_size)
    input: *mut f32,
) {
    // Integer ration between query heads count and kv heads count
    let kv_mul = head_count / kv_head_count;
    let kv_dim = (embedding_size * kv_head_count) / head_count;
    // For each attention head in the transformer
    for h_idx in 0..head_count {
        // Get the queries for this head
        let h_queries = unsafe { queries.add(h_idx * head_size) };
        // Get the attention scores for this head
        let h_attention_scores = unsafe { attention_scores.add(h_idx * seq_len) };
        // For each time step / position in the sequence length including this one
        for pos in 0..=current_position {
            let key_cache_offset = (layer * seq_len * kv_dim)
                + (pos * kv_dim)
                + ((h_idx / kv_mul) * head_size);
            let h_keys = unsafe { keys_cache.add(key_cache_offset) };
            let mut score = 0.0f32;
            // Compute the attention scores
            for head_pos in 0..head_size {
                unsafe {
                    score += *h_keys.add(head_pos) * *h_queries.add(head_pos)
                };
            }
            // Normalize and scale by the square root of the length
            score /= (head_size as f32).sqrt();
            unsafe { *h_attention_scores.add(pos) = score };
        }
        // Activate layer to get the attention weights, including current position
        unsafe { softmax(h_attention_scores, current_position+1) };

        // Go to the right position in the output values
        let h_weight_attention = unsafe { weigh_attention.add(h_idx * head_size) };
        // Prepare the buffer to accumulate weighted attention
        unsafe { h_weight_attention.write_bytes(0u8, head_size * core::mem::size_of::<f32>()); }
        // For each of the positions
        for pos in 0..=current_position {
            // Get the values from the value cache
            let h_values_cache_offset = (layer * seq_len * kv_dim)
                + (pos * kv_dim)
                + ((h_idx / kv_mul) * head_size);
            let h_values_cache = unsafe { values_cache.add(h_values_cache_offset) };
            let attention = unsafe { *h_attention_scores.add(pos) };
            // For each attention score in the entire head
            for head_pos in 0..head_size {
                // Accumulate the weighted attention scores
                unsafe { *h_weight_attention.add(head_pos) += attention
                    * *h_values_cache.add(head_pos) };
            }
        }
    }
    // Final activation with the output weights
    let weights = unsafe { output_weights.add(layer * embedding_size * embedding_size) };
    // or
    // let out = output_weights.add(layer * (head_size * head_count) * embedding_size);
    unsafe { matrix_mul(heads_activation, weigh_attention, weights, embedding_size, embedding_size) };

    // Also connect the residual branch
    for idx in 0..embedding_size {
        unsafe { *input.add(idx) += *heads_activation.add(idx) };
    }
}

#[repr(C)]
pub struct KVCache {
    keys: *const f32,
    values: *const f32,
}

/// Root mean squared normalization or RMSNorm is a layer optimisation technique proved to be
/// more efficient computation-wise than LayerNorm and it is used to normalise the layer by giving
/// the model re-scaling invariance property. It reduces the running time of a model by a factor
/// between 7% and 64% compared to LayerNorm. The following function computes RMS over `inputs`
/// and normalizes the activation by multiplying the `weights` with inputs and dividing by the RMS
/// previously computed.
/// # Safety
/// The results is returned through `out` pointer and all vectors have `size` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rms_norm(
    out: *mut f32,
    input: *const f32,
    weights: *const f32,
    size: usize,
) {
    let mut squared_sum = 0f32;
    // Compute the sum of all the input's squares
    for idx in 0..size {
        squared_sum += unsafe { *input.add(idx) * *input.add(idx) };
    }

    // Parameter to make sure the squared sum over size is not intepreted as zero.
    let epsilon = 1e-5;
    let rms: f32 = epsilon + (1.0f32 / size as f32 + squared_sum).sqrt();

    // Normalize / Activate the layer using rms
    for idx in 0..size {
        unsafe { *out.add(idx) = *input.add(idx) * *weights.add(idx) / rms };
    }
}

/// Given a certain `layer` move the `keys` and `value` pointers to the cache position for that
/// layer and that `position` in the sequence.
/// # Safety
/// The cache has the following dimension:
/// (layers, seq_len, embedding_size)
/// Which means that for each layer, we have the same number of key and value vectors to the number
/// of characters in the sequence. Each of which have the `embedding_size` which is the size of the
/// transformer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kv_cache(
    keys: *const f32,
    values: *const f32,
    layer: usize,
    seq_len: usize,
    kv_dim: usize,
    position: usize,
) -> KVCache {
    // Go to the desired layer
    let layer_cache = layer * (seq_len * kv_dim);
    // Go to the desired position
    let position_cache = layer_cache + position * kv_dim;

    unsafe {
        // Get pointer to the keys cache
        let keys = keys.add(position_cache);
        // Get pointer to the values cache
        let values = values.add(position_cache);

        KVCache { keys, values }
    }
}

/// Rotate position embeddings over the entire `embedding_size` space to the position `pos`.
/// The `head_size` represents the dimension of a single attention head and `kv_dim` is the
/// dimension of both the keys and the values tensors.
#[unsafe(no_mangle)]
pub extern "C" fn rope(
    embedding_size: usize,
    pos: usize,
    head_size: usize,
    kv_dim: usize,
    queries: *mut f32,
    keys: *mut f32,
) {
    // We rotate over the entire token embedding space, 2 tokens at a time
    for i in (0..embedding_size).step_by(2) {
        let head_dim = i % head_size;
        // Compute the angle by which the tokens have to be rotated. Usually this angle is computed
        // and scaled based on the full embeddings dimension, however here, we scale it based
        // on the position in the attention head
        let angle = 1.0f32 / 10000_f32.powf(head_dim as f32 / head_size as f32);
        // Move the angle to the required position
        let angle = pos as f32 * angle;
        // Compute the sin and cos values for the angle
        let cos = angle.cos();
        let sin = angle.sin();
        // Because we are using multiquery, it is the case that we might have more quries than
        // keys. We have to check for that and make sure we do not rotate invalid indexed keys.
        let q_and_k_to_rotate = if i < kv_dim {
            // Rotate queries and keys
            2
        } else {
            // Only rotate keys
            1
        };
        // This for loop is an easier way to write rotation for either only queries or both
        // queries and keys
        for vec_idx in 0..q_and_k_to_rotate {
            let to_rotate = if vec_idx == 0 { queries } else { keys };
            unsafe {
                // Get the 2 values to be rotated
                let v0 = *to_rotate.add(i);
                let v1 = *to_rotate.add(i + 1);
                // Rotate their values
                *to_rotate.add(i) = v0 * cos - v1 * sin;
                *to_rotate.add(i + 1) = v0 * sin + v1 * cos;
            }
        }
    }
}

// Multiply the contents of the input matrix with the contents of the weigth matrix and store the
// results in the `out` matrix.
/// # Safety
/// Make sure that the following dimensions match for the 3 vectors
/// 1. `input` is (size,), a 1-dimensional tensor
/// 2. `weights` is (dimensions, size), a 2-dimensional tensor
/// 3. `out` is W @ I (dimensions,), a 1-dimensional tensor
#[unsafe(no_mangle)]
pub unsafe extern "C" fn matrix_mul(
    out: *mut f32,
    input: *const f32,
    weights: *const f32,
    size: usize,
    dimensions: usize,
) {
    unsafe {
        for dim in 0..dimensions {
            let mut sum = 0f32;
            for idx in 0..size {
                // TODO: Is this a wrapping add and a wrapping mul actually?
                sum += *weights.add(dim * size + idx) * *input.add(idx);
            }
            *out.add(dim) = sum;
        }
    }
}

#[cfg(test)]
mod tests {}
