use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    qwen3_next::{
        QEmbedding, QLinear, QuantizationConfig, SwitchMlpWeights, new_mlp_projections, swiglu,
    },
    utils::{AttentionMask, create_attention_mask},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_norm_topk_prob() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Glm4MoeLiteModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    #[serde(default)]
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // MLA params
    pub kv_lora_rank: i32,
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub qk_nope_head_dim: i32,

    // MoE params
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub first_k_dense_replace: i32,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub num_nextn_predict_layers: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl Glm4MoeLiteModelArgs {
    const fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn is_moe_layer(&self, layer_idx: i32) -> bool {
        let Some(n_routed) = self.n_routed_experts else {
            return false;
        };
        if n_routed <= 0 {
            return false;
        }
        if layer_idx < self.first_k_dense_replace {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// RoPE helpers
// ---------------------------------------------------------------------------

fn apply_rope(x: &Array, dim: i32, base: f32, offset: i32) -> Result<Array, Exception> {
    mlx_rs::fast::rope(x, dim, false, base, 1.0, offset, None::<&Array>)
}

// ---------------------------------------------------------------------------
// MLA Attention (no YaRN)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Glm4MoeLiteAttention {
    #[param]
    q_a_proj: Option<QLinear>,
    #[param]
    q_a_layernorm: Option<nn::RmsNorm>,
    #[param]
    q_b_proj: Option<QLinear>,
    #[param]
    q_proj: Option<QLinear>,

    #[param]
    kv_a_proj_with_mqa: QLinear,
    #[param]
    kv_a_layernorm: nn::RmsNorm,
    #[param]
    kv_b_proj: QLinear,
    #[param]
    o_proj: QLinear,

    num_heads: i32,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    _v_head_dim: i32,
    scale: f32,
    rope_base: f32,
    use_compressed_query: bool,
}

impl Glm4MoeLiteAttention {
    fn new(args: &Glm4MoeLiteModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let q_head_dim = args.q_head_dim();
        let q_head_dim_f = f32::from(
            i16::try_from(q_head_dim)
                .map_err(|_| Exception::custom("q_head_dim out of i16 range"))?,
        );
        let scale = q_head_dim_f.sqrt().recip();

        let use_compressed_query = args.q_lora_rank.is_some();

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = args.q_lora_rank {
                (
                    Some(QLinear::new(ql, qb)?),
                    Some(
                        nn::RmsNormBuilder::new(q_lora_rank)
                            .eps(1e-6)
                            .build()?,
                    ),
                    Some(QLinear::new(ql, qb)?),
                    None,
                )
            } else {
                (None, None, None, Some(QLinear::new(ql, qb)?))
            };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa: QLinear::new(ql, qb)?,
            kv_a_layernorm: nn::RmsNormBuilder::new(args.kv_lora_rank)
                .eps(1e-6)
                .build()?,
            kv_b_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            num_heads: args.num_attention_heads,
            kv_lora_rank: args.kv_lora_rank,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            _v_head_dim: args.v_head_dim,
            scale,
            rope_base: args.rope_theta,
            use_compressed_query,
        })
    }

    #[allow(non_snake_case)]
    fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let q_projected = if self.use_compressed_query {
            let qa = self
                .q_a_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_a_proj missing"))?;
            let qa_ln = self
                .q_a_layernorm
                .as_mut()
                .ok_or_else(|| Exception::custom("q_a_layernorm missing"))?;
            let qb = self
                .q_b_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_b_proj missing"))?;
            qb.forward(&qa_ln.forward(&qa.forward(x)?)?)?
        } else {
            let qp = self
                .q_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_proj missing"))?;
            qp.forward(x)?
        };

        let q = q_projected
            .reshape(&[
                B,
                L,
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            ])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((.., .., .., ..self.qk_nope_head_dim));
        let q_pe_raw = q.index((.., .., .., self.qk_nope_head_dim..));

        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_latent = compressed_kv.index((.., .., ..self.kv_lora_rank));
        let k_pe_raw = compressed_kv
            .index((.., .., self.kv_lora_rank..))
            .reshape(&[B, L, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&kv_latent)?)?
            .reshape(&[B, L, self.num_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k_nope = kv.index((.., .., .., ..self.qk_nope_head_dim));
        let v_decompressed = kv.index((.., .., .., self.qk_nope_head_dim..));

        let offset = cache.as_ref().map_or(0, |c| KeyValueCache::offset(*c));

        let q_pe = apply_rope(&q_pe_raw, self.qk_rope_head_dim, self.rope_base, offset)?;
        let k_pe = apply_rope(&k_pe_raw, self.qk_rope_head_dim, self.rope_base, offset)?;

        let k_pe_expanded =
            ops::broadcast_to(&k_pe, &[B, self.num_heads, L, self.qk_rope_head_dim])?;

        let keys_combined = ops::concatenate_axis(&[&k_nope, &k_pe_expanded], -1)?;
        let queries = ops::concatenate_axis(&[&q_nope, &q_pe], -1)?;

        let (keys, values) = if let Some(kv_cache) = cache {
            kv_cache.update_and_fetch(keys_combined, v_decompressed)?
        } else {
            (keys_combined, v_decompressed)
        };

        let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
        let output = fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            self.scale,
            sdpa_mask,
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }
}

// ---------------------------------------------------------------------------
// Shared experts MLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SharedExperts {
    #[param]
    gate_proj: QLinear,
    #[param]
    down_proj: QLinear,
    #[param]
    up_proj: QLinear,
}

impl SharedExperts {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(ql, qb)?,
            down_proj: QLinear::new(ql, qb)?,
            up_proj: QLinear::new(ql, qb)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let activated = swiglu(&self.gate_proj.forward(x)?, &self.up_proj.forward(x)?)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MLP block (dense or sparse MoE with noaux_tc gating)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Glm4MoeLiteMlpBlock {
    #[param]
    gate: Option<nn::Linear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    #[param]
    shared_experts: Option<SharedExperts>,
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,

    num_experts: i32,
    top_k: i32,
    scaling_factor: f32,
    norm_topk_prob: bool,
    is_moe: bool,
}

impl Glm4MoeLiteMlpBlock {
    fn new_moe(args: &Glm4MoeLiteModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n_routed = args
            .n_routed_experts
            .ok_or_else(|| Exception::custom("n_routed_experts required for MoE layer"))?;
        let top_k = args.num_experts_per_tok.unwrap_or(2);

        let shared = if args.n_shared_experts.is_some_and(|n| n > 0) {
            Some(SharedExperts::new(ql, qb)?)
        } else {
            None
        };

        Ok(Self {
            gate: Some(
                nn::LinearBuilder::new(args.hidden_size, n_routed)
                    .bias(false)
                    .build()?,
            ),
            switch_mlp: Some(SwitchMlpWeights::new(ql, qb)?),
            shared_experts: shared,
            gate_proj: None,
            down_proj: None,
            up_proj: None,
            num_experts: n_routed,
            top_k,
            scaling_factor: args.routed_scaling_factor.unwrap_or(1.0),
            norm_topk_prob: args.norm_topk_prob,
            is_moe: true,
        })
    }

    fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_experts: None,
            gate_proj: Some(gate_proj),
            down_proj: Some(down_proj),
            up_proj: Some(up_proj),
            num_experts: 0,
            top_k: 0,
            scaling_factor: 1.0,
            norm_topk_prob: false,
            is_moe: false,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            self.forward_moe(x)
        } else {
            self.forward_dense(x)
        }
    }

    fn forward_moe(&mut self, x: &Array) -> Result<Array, Exception> {
        let gates_raw = self
            .gate
            .as_mut()
            .ok_or_else(|| Exception::custom("MoE router gate missing"))?
            .forward(x)?;
        let gates = ops::softmax_axis(&gates_raw, -1, true)?;

        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let top_k_start = self.num_experts - self.top_k;
        let top_inds = all_inds.index((.., .., top_k_start..));
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        // noaux_tc: normalize top-k scores to sum to 1, then apply scaling factor
        let normalized = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        let scaled_scores = if (self.scaling_factor - 1.0).abs() > f32::EPSILON {
            let scalar = Array::from_f32(self.scaling_factor).as_dtype(normalized.dtype())?;
            normalized.multiply(&scalar)?
        } else {
            normalized
        };

        let y = self
            .switch_mlp
            .as_ref()
            .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?
            .forward_gather_global_sort(x, &top_inds)?;
        let mut result = y
            .multiply(&scaled_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)?;

        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            result = result.add(shared_out)?;
        }

        Ok(result)
    }

    fn forward_dense(&self, x: &Array) -> Result<Array, Exception> {
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let dp = self
            .down_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;

        let activated = swiglu(&gp.forward(x)?, &up.forward(x)?)?;
        dp.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Glm4MoeLiteDecoderLayer {
    #[param]
    self_attn: Glm4MoeLiteAttention,
    #[param]
    mlp: Glm4MoeLiteMlpBlock,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl Glm4MoeLiteDecoderLayer {
    fn new(
        args: &Glm4MoeLiteModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
        let mlp = if args.is_moe_layer(layer_idx) {
            Glm4MoeLiteMlpBlock::new_moe(args, ql, qb)?
        } else {
            Glm4MoeLiteMlpBlock::new_dense(ql, qb)?
        };

        Ok(Self {
            self_attn: Glm4MoeLiteAttention::new(args, ql, qb)?,
            mlp,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(attn_out)?;

        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Inner model (embed + layers + norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Glm4MoeLiteInner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<Glm4MoeLiteDecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl Glm4MoeLiteInner {
    fn new(args: &Glm4MoeLiteModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| Glm4MoeLiteDecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Glm4MoeLiteCausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Glm4MoeLiteCausalLM {
    pub args: Glm4MoeLiteModelArgs,
    #[param]
    model: Glm4MoeLiteInner,
    #[param]
    lm_head: Option<QLinear>,
}

impl Glm4MoeLiteCausalLM {
    pub fn new(args: Glm4MoeLiteModelArgs) -> Result<Self, Exception> {
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_attention_heads.is_positive() {
            return Err(Exception::custom("num_attention_heads must be positive"));
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = Glm4MoeLiteInner::new(&args, ql, qb)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(ql, qb)?)
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        let computed_mask = match mask {
            Some(m) => Some(AttentionMask::Array(m.clone())),
            None => create_attention_mask(&h, kv_cache, None)?,
        };

        if kv_cache.is_empty() {
            *kv_cache = (0..self.model.layers.len())
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
        } else if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            h = layer.forward(&h, computed_mask.as_ref(), layer_cache.as_mut())?;
        }

        self.model.norm.forward(&h)
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        let h_last = h.index((.., -1.., ..));

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h_last),
            None => self.model.embed_tokens.as_linear(&h_last),
        }
    }

    #[allow(non_snake_case)]
    pub fn forward_all_logits(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h),
            None => self.model.embed_tokens.as_linear(&h),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Glm4MoeLiteModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_glm4_moe_lite_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Glm4MoeLiteCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        kv_lora_rank = args.kv_lora_rank,
        q_lora_rank = ?args.q_lora_rank,
        n_routed_experts = ?args.n_routed_experts,
        n_shared_experts = ?args.n_shared_experts,
        vocab_size = args.vocab_size,
        "Loading glm4_moe_lite model"
    );

    let mut model = Glm4MoeLiteCausalLM::new(args)?;

    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("Glm4MoeLite model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn glm4_moe_lite_args() -> Glm4MoeLiteModelArgs {
        Glm4MoeLiteModelArgs {
            model_type: "glm4_moe_lite".to_owned(),
            hidden_size: 2048,
            num_hidden_layers: 47,
            intermediate_size: 1536,
            num_attention_heads: 20,
            num_key_value_heads: 20,
            rms_norm_eps: 1e-6,
            vocab_size: 151_936,
            max_position_embeddings: 131_072,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            kv_lora_rank: 512,
            q_lora_rank: Some(768),
            qk_rope_head_dim: 64,
            v_head_dim: 256,
            qk_nope_head_dim: 192,
            n_routed_experts: Some(64),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(4),
            moe_intermediate_size: Some(1536),
            routed_scaling_factor: Some(1.8),
            first_k_dense_replace: 1,
            norm_topk_prob: true,
            num_nextn_predict_layers: 1,
            quantization: Some(QuantizationConfig {
                group_size: 64,
                bits: 4,
            }),
        }
    }

    fn small_args() -> Glm4MoeLiteModelArgs {
        Glm4MoeLiteModelArgs {
            model_type: "glm4_moe_lite".to_owned(),
            hidden_size: 64,
            num_hidden_layers: 2,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 128,
            max_position_embeddings: 256,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            kv_lora_rank: 32,
            q_lora_rank: Some(16),
            qk_rope_head_dim: 8,
            v_head_dim: 16,
            qk_nope_head_dim: 16,
            n_routed_experts: Some(4),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(2),
            moe_intermediate_size: Some(32),
            routed_scaling_factor: Some(1.8),
            first_k_dense_replace: 1,
            norm_topk_prob: true,
            num_nextn_predict_layers: 1,
            quantization: None,
        }
    }

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "glm4_moe_lite",
            "hidden_size": 2048,
            "num_hidden_layers": 47,
            "intermediate_size": 1536,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "kv_lora_rank": 512,
            "q_lora_rank": 768,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "qk_nope_head_dim": 192,
            "n_routed_experts": 64,
            "n_shared_experts": 1,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1536,
            "routed_scaling_factor": 1.8,
            "first_k_dense_replace": 1,
            "norm_topk_prob": true,
            "num_nextn_predict_layers": 1
        }"#;

        let args: Glm4MoeLiteModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "glm4_moe_lite");
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.kv_lora_rank, 512);
        assert_eq!(args.q_lora_rank, Some(768));
        assert_eq!(args.qk_nope_head_dim, 192);
        assert_eq!(args.qk_rope_head_dim, 64);
        assert_eq!(args.v_head_dim, 256);
        assert_eq!(args.q_head_dim(), 256);
        assert_eq!(args.n_routed_experts, Some(64));
        assert_eq!(args.n_shared_experts, Some(1));
        assert_eq!(args.num_experts_per_tok, Some(4));
        assert_eq!(args.routed_scaling_factor, Some(1.8));
        assert!(args.norm_topk_prob);
    }

    #[test]
    fn test_config_defaults() {
        let json = r#"{
            "model_type": "glm4_moe_lite",
            "hidden_size": 2048,
            "num_hidden_layers": 47,
            "intermediate_size": 1536,
            "num_attention_heads": 20,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "qk_nope_head_dim": 192
        }"#;

        let args: Glm4MoeLiteModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.rope_theta, 1_000_000.0);
        assert_eq!(args.max_position_embeddings, 0);
        assert!(!args.tie_word_embeddings);
        assert!(args.q_lora_rank.is_none());
        assert!(args.n_routed_experts.is_none());
        assert!(args.n_shared_experts.is_none());
        assert!(!args.norm_topk_prob);
        assert_eq!(args.first_k_dense_replace, 0);
    }

    #[test]
    fn test_q_head_dim() {
        let args = glm4_moe_lite_args();
        assert_eq!(args.q_head_dim(), 256);
    }

    #[test]
    fn test_is_moe_layer_first_dense() {
        let args = glm4_moe_lite_args();
        assert!(
            !args.is_moe_layer(0),
            "layer 0 should be dense (first_k_dense_replace=1)"
        );
    }

    #[test]
    fn test_is_moe_layer_after_dense() {
        let args = glm4_moe_lite_args();
        assert!(args.is_moe_layer(1));
        assert!(args.is_moe_layer(10));
        assert!(args.is_moe_layer(46));
    }

    #[test]
    fn test_is_moe_layer_no_experts() {
        let mut args = glm4_moe_lite_args();
        args.n_routed_experts = None;
        assert!(!args.is_moe_layer(5));
    }

    #[test]
    fn test_model_new_small_config() {
        let args = small_args();
        let model = Glm4MoeLiteCausalLM::new(args).unwrap();
        assert_eq!(model.args.num_hidden_layers, 2);
        assert!(model.lm_head.is_none(), "tied embeddings => no lm_head");
    }

    #[test]
    fn test_model_new_untied_embeddings() {
        let mut args = small_args();
        args.tie_word_embeddings = false;
        let model = Glm4MoeLiteCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_model_new_zero_layers() {
        let mut args = small_args();
        args.num_hidden_layers = 0;
        assert!(Glm4MoeLiteCausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_zero_vocab() {
        let mut args = small_args();
        args.vocab_size = 0;
        assert!(Glm4MoeLiteCausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_zero_attention_heads() {
        let mut args = small_args();
        args.num_attention_heads = 0;
        assert!(Glm4MoeLiteCausalLM::new(args).is_err());
    }

    #[test]
    fn test_forward_preserves_pre_initialized_cache() {
        let args = small_args();
        let mut model = Glm4MoeLiteCausalLM::new(args).unwrap();
        let mut cache: Vec<Option<SteppingKeyValueCache>> =
            (0..2).map(|_| Some(SteppingKeyValueCache::new())).collect();

        let input = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        let _ = model.forward(&input, None, &mut cache);
        assert_eq!(cache.len(), 2);
        for (i, c) in cache.iter().enumerate() {
            assert!(c.is_some(), "layer {i} cache should be Some");
        }
    }

    #[test]
    fn test_forward_cache_length_mismatch() {
        let args = small_args();
        let mut model = Glm4MoeLiteCausalLM::new(args).unwrap();
        let mut cache = vec![
            Some(SteppingKeyValueCache::new()),
            Some(SteppingKeyValueCache::new()),
            Some(SteppingKeyValueCache::new()),
        ];

        let input = Array::from_slice(&[1_i32], &[1, 1]);
        let result = model.forward(&input, None, &mut cache);
        assert!(result.is_err(), "mismatched cache length should error");
    }

    #[test]
    fn test_load_model_args_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_model_args(dir.path()).is_err());
    }

    #[test]
    fn test_load_model_args_invalid_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        assert!(load_model_args(dir.path()).is_err());
    }
}
