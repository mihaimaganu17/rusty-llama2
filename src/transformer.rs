use crate::read::Reader;
use std::rc::Rc;

#[repr(C)]
pub struct Transformer {
    // The configuration we read from the file
    pub config: Config,
    // The weights of the model
    pub weights: Weights,
    // All the buffers needed for one forward pass through the entire model
    pub state: State,
}

impl Transformer {
    pub fn from_reader(reader: &mut Reader) -> Result<Self, Error> {
        let mut config = Config::from_reader(reader)?;
        // Negative vocabulary size is the way of signaling unshared weights.
        let shared_weights = if config.vocab_size > 0 { true } else { false };
        // Convert vocabulary size to it's positive value
        config.vocab_size = config.vocab_size.abs();

        let _weights = WeightsSafe::from_reader(reader, &config, shared_weights)?;
        Err(Error::Bad)
    }
}

#[repr(C)]
pub struct Config {
    // Tranformer dimension and the size occupied by the embedding for each token
    embedding_size: u32,
    // Dimension for the hidden layer of the FFN network which cumullates two projections
    hidden_dim: u32,
    // Total number of layers in the transformer
    layer_count: u32,
    // Number of attention/query heads in the transformer
    heads_count: u32,
    // Number of key/value heads. Can be less than query heads because of multiquery
    kv_heads_count: u32,
    // Vocabulary size
    vocab_size: i32,
    // Maximum sequence length
    seq_len: u32,
}

impl Config {
    pub fn from_reader(reader: &mut Reader) -> Result<Self, Error> {
        let config = Config {
            embedding_size: reader.read_u32()?,
            hidden_dim: reader.read_u32()?,
            layer_count: reader.read_u32()?,
            heads_count: reader.read_u32()?,
            kv_heads_count: reader.read_u32()?,
            vocab_size: reader.read_i32()?,
            seq_len: reader.read_u32()?,
        };


        Ok(config)
    }

    // Tranformer dimension and the size occupied by the embedding for each token
    pub fn embedding_size(&self) -> u32 {
        self.embedding_size
    }

    // Dimension for the hidden layer of the FFN network which cumullates two projections
    pub fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    // Total number of layers in the transformer
    pub fn layer_count(&self) -> u32 {
        self.layer_count
    }

    // Number of attention/query heads in the transformer
    pub fn heads_count(&self) -> u32 {
        self.heads_count
    }

    // Number of key/value heads. Can be less than query heads because of multiquery
    pub fn kv_heads_count(&self) -> u32 {
        self.kv_heads_count
    }

    // Vocabulary size
    pub fn vocab_size(&self) -> i32 {
        self.vocab_size
    }

    // Maximum sequence length
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }
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
    w_projection_activation: *const f32,
    w_projection2: *const f32,

    // Final RMS norm, before logits
    w_rms_final: *const f32,
    // (optional) classifier weights for the logits, on the last layer
    w_cls: *const f32,
}

impl Weights {
    // Token embedding table (vocab_size, embedding_size)
    pub fn token_embedding_table(&self) -> *const f32 {
        self.token_embedding_table
    }

    // Weights for RMS norms, each with (layer_count, embedding_size) size
    pub fn w_rms_att(&self) -> *const f32 {
        self.w_rms_att
    }

    pub fn w_rms_ffn(&self) -> *const f32 {
        self.w_rms_ffn
    }

    // Weights for attention
    // Queries: (layer_count, embedding_size, heads_count * head_size)
    pub fn w_queries(&self) -> *const f32 {
        self.w_queries
    }

    // Keys: (layer_count, embedding_size, kv_heads_count * head_size)
    pub fn w_keys(&self) -> *const f32 {
        self.w_keys
    }

    // Values: (layer_count, embedding_size, kv_heads_count * head_size)
    pub fn w_values(&self) -> *const f32 {
        self.w_values
    }

    // Attentions scores output weights: (layer_count, heads_count * head_size, embedding_size)
    pub fn w_att_out(&self) -> *const f32 {
        self.w_att_out
    }

    // Weights for FFN
    pub fn w_projection1(&self) -> *const f32 {
        self.w_projection1
    }

    pub fn w_projection2(&self) -> *const f32 {
        self.w_projection2
    }

    pub fn w_projection_activation(&self) -> *const f32 {
        self.w_projection_activation
    }

    // Final RMS norm, before logits
    pub fn w_rms_final(&self) -> *const f32 {
        self.w_rms_final
    }

    // (optional) classifier weights for the logits, on the last layer
    pub fn w_cls(&self) -> *const f32 {
        self.w_cls
    }
}

// Same as `Weights` but for safe Rust
#[repr(C)]
pub struct WeightsSafe {
    // Token embedding table (vocab_size, embedding_size)
    token_embedding_table: Rc<Vec<f32>>,
    // Weights for RMS norms, each with (layer_count, embedding_size) size
    w_rms_att: Vec<f32>,
    w_rms_ffn: Vec<f32>,
    // Weights for attention
    // Queries: (layer_count, embedding_size, heads_count * head_size)
    w_queries: Vec<f32>,
    // Keys: (layer_count, embedding_size, kv_heads_count * head_size)
    w_keys: Vec<f32>,
    // Values: (layer_count, embedding_size, kv_heads_count * head_size)
    w_values: Vec<f32>,
    // Attentions scores output weights: (layer_count, heads_count * head_size, embedding_size)
    w_att_out: Vec<f32>,

    // Weights for FFN
    w_projection1: Vec<f32>,
    w_projection_activation: Vec<f32>,
    w_projection2: Vec<f32>,

    // Final RMS norm, before logits
    w_rms_final: Vec<f32>,
    // (optional) classifier weights for the logits, on the last layer
    w_cls: Rc<Vec<f32>>,
}

impl WeightsSafe {
    pub fn from_reader(reader: &mut Reader, config: &Config, shared_weights: bool) -> Result<Self, Error> {
        let head_size = config.embedding_size / config.heads_count;
        let count = config.embedding_size * config.vocab_size as u32;
        let token_embedding_table = Rc::new((0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>());
        assert_eq!(token_embedding_table.len(), count as usize);
        let count = config.layer_count * config.embedding_size;
        let w_rms_att = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_rms_att.len(), count as usize);
        let count = config.layer_count * config.embedding_size;
        let w_rms_ffn = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_rms_ffn.len(), count as usize);
        let count = config.layer_count * config.embedding_size * config.heads_count * head_size;
        let w_queries = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_queries.len(), count as usize);
        let count = config.layer_count * config.embedding_size * config.kv_heads_count * head_size;
        let w_keys= (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_keys.len(), count as usize);
        let count = config.layer_count * config.embedding_size * config.kv_heads_count * head_size;
        let w_values = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_values.len(), count as usize);
        let count = config.layer_count * config.heads_count * head_size * config.embedding_size;
        let w_att_out = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_att_out.len(), count as usize);
        let count = config.layer_count * config.embedding_size * config.hidden_dim;
        let w_projection1 = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_projection1.len(), count as usize);
        let count = config.layer_count * config.hidden_dim * config.embedding_size;
        let w_projection_activation = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_projection_activation.len(), count as usize);
        let count = config.layer_count * config.embedding_size * config.hidden_dim;
        let w_projection2 = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_projection2.len(), count as usize);
        let count = config.embedding_size;
        let w_rms_final = (0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>();
        assert_eq!(w_rms_final.len(), count as usize);
        // Skip what used to be freq_cis_real (RoPE) and freq_cis_imag(RoPE)
        let count = config.seq_len * head_size;
        let _ = (0..count).map(|_| reader.read_f32());


        let count = config.embedding_size* config.vocab_size as u32;
        let w_cls = if shared_weights {
            token_embedding_table.clone()
        } else {
            Rc::new((0..count)
            .map(|_| reader.read_f32()).flatten().collect::<Vec<_>>())
        };
        assert_eq!(w_cls.len(), count as usize);
        Ok({
            Self {
                token_embedding_table,
                w_rms_att,
                w_rms_ffn,
                w_queries,
                w_keys,
                w_values,
                w_att_out,
                w_projection1,
                w_projection_activation,
                w_projection2,
                w_rms_final,
                w_cls,
            }
        })
    }
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
    //cache: KVCache,
    pub cache_keys: *const f32,
    pub cache_values: *const f32,
}

impl State {
    // Activation at the current token (embedding_size,)
    pub fn token_emb(&self) -> *const f32 {
        self.token_emb
    }

    // Same embedding but connected with the residual branch (embedding_size,)
    pub fn token_emb_res(&self) -> *const f32 {
        self.token_emb_res
    }

    // Additional buffer for convenience
    pub fn temp_buffer(&self) -> *const f32 {
        self.temp_buffer
    }

    // Buffers for the 2 projections of the FFN's hidden layer, both (hidden_dim,)
    pub fn hidden_buffer1(&self) -> *const f32 {
        self.hidden_buffer1
    }

    pub fn hidden_buffer2(&self) -> *const f32 {
        self.hidden_buffer2
    }

    // Buffers to hold the results for queries, keys, values, all of size (embedding_size,)
    pub fn queries(&self) -> *const f32 {
        self.queries
    }

    pub fn keys(&self) -> *const f32 {
        self.keys
    }

    pub fn values(&self) -> *const f32 {
        self.values
    }

    // Buffer for attention scores (heads_count, seq_len)
    pub fn att_scores(&self) -> *const f32 {
        self.att_scores
    }

    // Output logits (embedding_size, vocab_size)
    pub fn logits(&self) -> *const f32 {
        self.logits
    }

    // Key - Value cache
    //pub fn cache(&self) -> &KVCache {
    //    &self.cache
    //}

    /// Given a certain `layer` move the `keys` and `value` pointers to the cache position for that
    /// layer and that `position` in the sequence.
    /// # Safety
    /// The cache has the following dimension:
    /// (layers, seq_len, embedding_size)
    /// Which means that for each layer, we have the same number of key and value vectors to the number
    /// of characters in the sequence. Each of which have the `embedding_size` which is the size of the
    /// transformer.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn layer_cache(
        &mut self,
        layer: u32,
        seq_len: u32,
        kv_dim: u32,
        position: u32,
    ) {
        // Go to the desired layer
        let layer_cache = layer * (seq_len * kv_dim);
        // Go to the desired position
        let position_cache = layer_cache + position * kv_dim;

        unsafe {
            // Get pointer to the keys cache
            self.keys = self.cache_keys.add(position_cache as usize);
            // Get pointer to the values cache
            self.values = self.cache_values.add(position_cache as usize);
        }
    }
}

pub enum Error {
    Bad,
    Reader(crate::read::Error),
}

impl From<crate::read::Error> for Error {
    fn from(err: crate::read::Error) -> Self {
        Self::Reader(err)
    }
}
