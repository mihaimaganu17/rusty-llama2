#[unsafe(no_mangle)]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust!");
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
