#[unsafe(no_mangle)]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust!");
}

/// Rotate position embeddings over the entire `embedding_size` space to the position `pos`.
/// The `head_size` represents the dimension of a single attention head and `kv_dim` is the
/// dimension of both the keys and the values tensors.
#[unsafe(no_mangle)]
pub extern "C" fn rope(embedding_size: usize, pos: usize, head_size: usize, kv_dim: usize,
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
            let to_rotate = if vec_idx == 0 {
                queries
            } else {
                keys
            };
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
// results in the `out` matrix. Shapes of the tensors are the following:
// 1. `input` is (size,), a 1-dimensional tensor
// 2. `weights` is (dimensions, size), a 2-dimensional tensor
// 3. `out` is W @ I (dimensions,), a 1-dimensional tensor
#[unsafe(no_mangle)]
pub extern "C" fn matrix_mul(
    out: *mut f32,
    input: *const f32,
    weights: *const f32,
    size: isize,
    dimensions: isize,
) {
    unsafe {
        for dim in 0..dimensions {
            let mut sum = 0f32;
            for idx in 0..size {
                // TODO: Is this a wrapping add and a wrapping mul actually?
                sum += *weights.offset(dim * size + idx) * *input.offset(idx);
            }
            *out.offset(dim) = sum;
        }
    }
}

#[cfg(test)]
mod tests {
}
