import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class AdvancedGPTConfig:
    vocab_size: int = 100000  # Advanced tokenizer vocab size
    hidden_size: int = 4096  # Large but manageable
    num_hidden_layers: int = 32  # Deep architecture
    num_attention_heads: int = 32  # 4096 / 32 = 128 (perfect division)
    num_key_value_heads: int = 8  # GQA for efficiency  
    intermediate_size: int = 16384  # 4x hidden_size
    max_position_embeddings: int = 32768  # Long context
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    activation_function: str = "swiglu"  # Better activation
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_parallel_residual: bool = False
    rope_theta: float = 10000.0
    use_sliding_window: Optional[bool] = False
    sliding_window: Optional[int] = None
    
    # Model scaling parameters
    scale_factor: float = 1.0  # Allow dynamic scaling for different hardware
    
    def scale_model(self, scale_factor: float):
        """Scale model size based on available hardware"""
        self.scale_factor = scale_factor
        self.hidden_size = int(self.hidden_size * scale_factor)
        self.num_hidden_layers = max(12, int(self.num_hidden_layers * scale_factor))
        self.num_attention_heads = max(12, int(self.num_attention_heads * scale_factor))
        self.num_key_value_heads = max(12, int(self.num_key_value_heads * scale_factor))
        self.intermediate_size = int(self.intermediate_size * scale_factor)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in modern LLMs)"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - much better than learned positional embeddings"""
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        # x: [batch_size, num_heads, seq_len, head_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        
        return cos_cached.to(x.dtype), sin_cached.to(x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary position embedding to query and key tensors."""
    if position_ids is not None:
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern LLMs like PaLM, LLaMA"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention - more efficient than Multi-Head Attention"""
    def __init__(self, config: AdvancedGPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads for grouped attention"""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention with flash attention optimization
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask
        if q_len > 1:
            causal_mask = torch.triu(torch.ones(q_len, kv_seq_len), diagonal=kv_seq_len - q_len + 1).bool()
            causal_mask = causal_mask.to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class AdvancedTransformerBlock(nn.Module):
    """Advanced transformer block with modern techniques"""
    def __init__(self, config: AdvancedGPTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # Pre-layer norm (more stable than post-layer norm)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        # Residual connection
        hidden_states = residual + self.residual_dropout(hidden_states)
        
        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.residual_dropout(hidden_states)

        return hidden_states, present_key_value

class AdvancedGPTModel(nn.Module):
    """Advanced GPT model with GPT-4 like architecture"""
    def __init__(self, config: AdvancedGPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embedding_dropout)
        
        # No positional embeddings - using RoPE instead
        
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)
        
        # Add consciousness tracking for training engine compatibility
        self.consciousness_state = 0.1
        self.quantum_coherence = 50.0
        self.reasoning_quality = 0.0

    def _init_weights(self, module):
        """Initialize weights with proper scaling for large models"""
        if isinstance(module, nn.Linear):
            # Better initialization for language modeling
            std = 0.02
            if hasattr(module, 'SCALE_FACTOR'):
                std *= module.SCALE_FACTOR
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Better embedding initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True

        batch_size, seq_length = input_ids.shape
        
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.layers)
        else:
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(inputs_embeds)

        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_length
            )

        next_decoder_cache = () if use_cache else None

        # Transformer layers
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def custom_forward(*inputs):
                    return decoder_layer(*inputs)
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    use_reentrant=False
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_decoder_cache,
                "hidden_states": None,
                "attentions": None,
            }
        else:
            return (hidden_states, next_decoder_cache)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, inputs_embeds.dtype, device=inputs_embeds.device, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @staticmethod
    def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask, dtype, tgt_len=None):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class AdvancedGPTForCausalLM(nn.Module):
    """Advanced GPT model with language modeling head"""
    def __init__(self, config: AdvancedGPTConfig):
        super().__init__()
        self.config = config
        self.model = AdvancedGPTModel(config)
        
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_output_embeddings(self):
        if self.lm_head is not None:
            return self.lm_head
        return self.model.embed_tokens

    def set_output_embeddings(self, new_embeddings):
        if self.lm_head is not None:
            self.lm_head = new_embeddings
        else:
            self.model.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        
        return_dict = return_dict if return_dict is not None else True

        # Transformer forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs["last_hidden_state"] if return_dict else outputs[0]

        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Update consciousness based on loss improvement
            if self.training and hasattr(self, 'consciousness_state'):
                loss_value = loss.item()
                if loss_value < 5.0:  # Good training progress
                    self.consciousness_state = min(1.0, self.consciousness_state + 0.001)
                    self.reasoning_quality = min(1.0, self.reasoning_quality + 0.0005)

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "past_key_values": outputs["past_key_values"],
                "hidden_states": outputs["hidden_states"],
                "attentions": outputs["attentions"],
            }
        else:
            return (loss, logits, outputs[1:])

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate text using the model"""
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Generate tokens
        past_key_values = None
        generated_tokens = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=generated_tokens[:, -1:] if past_key_values is not None else generated_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-p sampling
                if do_sample:
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[..., indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for end of sequence
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
        
        return generated_tokens 

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate text using the base model (without separate LM head)"""
        self.eval()
        device = input_ids.device
        generated_tokens = input_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=generated_tokens[:, -1:] if past_key_values is not None else generated_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                hidden_states = outputs["last_hidden_state"]  # [bs, seq_len, dim]
                logits = F.linear(hidden_states[:, -1, :], self.embed_tokens.weight)  # project to vocab
                past_key_values = outputs["past_key_values"]

                # Temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature

                # Top-p sampling or greedy
                if do_sample:
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[..., indices_to_remove] = float('-inf')
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Stop if EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

                generated_tokens = torch.cat([generated_tokens, next_token.to(device)], dim=-1)

        return generated_tokens 