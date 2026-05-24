"""
Mini-BLIP2 模型定义。
核心架构：Frozen CLIP ViT → Mini Q-Former (trainable) → Projection → Frozen OPT
"""

import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, OPTModel


class QFormerLayer(nn.Module):
    """单层 Q-Former：Self-Attention → Cross-Attention → FFN。"""

    def __init__(self, hidden_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, image_features):
        # Self-attention among queries
        residual = queries
        x, _ = self.self_attn(queries, queries, queries)
        x = self.dropout(x)
        queries = self.norm1(residual + x)

        # Cross-attention: queries → image features
        residual = queries
        x, _ = self.cross_attn(queries, image_features, image_features)
        x = self.dropout(x)
        queries = self.norm2(residual + x)

        # Feed-forward
        residual = queries
        x = self.ffn(queries)
        x = self.dropout(x)
        queries = self.norm3(residual + x)

        return queries


class MiniQFormer(nn.Module):
    """Mini Q-Former：learnable queries + 若干层 self/cross-attention。"""

    def __init__(self, num_queries=16, hidden_size=512, num_layers=2,
                 num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size

        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size) * 0.02)

        self.layers = nn.ModuleList([
            QFormerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, image_features):
        B = image_features.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, image_features)

        return queries


class MiniBLIP2(nn.Module):
    """Mini-BLIP2：Vision → Q-Former → Projection → Language → Caption。"""

    def __init__(self, vision_model_name=None,
                 language_model_name=None,
                 num_queries=16, qformer_hidden=512, qformer_layers=2,
                 qformer_heads=8):
        super().__init__()

        # 默认使用本地模型路径
        base = os.path.join(os.path.dirname(__file__), "..", "models")
        if vision_model_name is None:
            vision_model_name = os.path.join(base, "clip-vit-base-patch32")
        if language_model_name is None:
            language_model_name = os.path.join(base, "opt-125m")

        # ---- 视觉编码器（冻结） ----
        print(f"Loading vision encoder: {vision_model_name} ...")
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name, torch_dtype=torch.float32
        )
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        vision_hidden = self.vision_encoder.config.hidden_size  # 768 → wait, ViT-B/32 actually outputs 768!
        # Actually, CLIP ViT-B/32 hidden_size is 768, not 512. Let me handle this dynamically.
        # Hmm, let me check. The CLIP ViT-B/32 model from openai has hidden_size=768.
        # But I set qformer_hidden to 512. I need to align these.
        # Let me set qformer_hidden to 768 to match, or add a linear projection.

        # ---- 语言解码器（冻结） ----
        print(f"Loading language model: {language_model_name} ...")
        self.language_model = OPTModel.from_pretrained(
            language_model_name, torch_dtype=torch.float32
        )
        for p in self.language_model.parameters():
            p.requires_grad = False
        lm_hidden = self.language_model.config.hidden_size  # 768
        lm_vocab = self.language_model.config.vocab_size  # 50272

        # ---- 对齐 vision hidden → qformer hidden ----
        self.vision_proj = nn.Linear(vision_hidden, qformer_hidden)

        # ---- Mini Q-Former ----
        self.qformer = MiniQFormer(
            num_queries=num_queries,
            hidden_size=qformer_hidden,
            num_layers=qformer_layers,
            num_heads=qformer_heads,
        )

        # ---- Projection: qformer → language model ----
        self.projection = nn.Linear(qformer_hidden, lm_hidden)

        # ---- 输出头：lm_hidden → vocab（用于训练时的 loss 计算） ----
        # OPT 内部的 lm_head 在 OPTModel 中默认不输出 logits
        # 需要借助 lm_head 把 hidden states 映射到 vocab
        # OPTForCausalLM 才有 lm_head，但 OPTModel 没有。
        # 解决办法：用 language_model 的 inputs_embeds 方式训练
        # （直接在 forward 中拼接 prefix 和 text embeddings）

        print(f"Vision hidden: {vision_hidden}, Q-Former hidden: {qformer_hidden}, "
              f"LM hidden: {lm_hidden}, Vocab: {lm_vocab}")

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        B = pixel_values.shape[0]
        device = pixel_values.device

        # 1. Frozen vision encoder
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values, output_hidden_states=True)
            # Use last hidden state: (B, 50, vision_hidden)
            # CLIP ViT has 50 patches (1 CLS + 49 patches)
            image_features = vision_out.last_hidden_state

        # 1.5. Project vision features to qformer hidden size
        image_features = self.vision_proj(image_features)

        # 2. Q-Former
        qformer_out = self.qformer(image_features)  # (B, num_queries, qformer_hidden)

        # 3. Projection to language space
        prefix_embeds = self.projection(qformer_out)  # (B, num_queries, lm_hidden)

        # 4. Get text embeddings from frozen language model
        embed_layer = self.language_model.get_input_embeddings()
        text_embeds = embed_layer(input_ids)  # (B, L, lm_hidden)

        # Pad text embeddings to a multiple of 2 for OPT compatibility
        # OPT requires input length to be multiple of 2 in some configurations
        current_len = prefix_embeds.shape[1] + text_embeds.shape[1]
        if current_len % 2 != 0:
            text_embeds = torch.cat([
                text_embeds,
                torch.zeros(B, 1, text_embeds.shape[2], device=device, dtype=text_embeds.dtype)
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(B, 1, device=device, dtype=attention_mask.dtype)
            ], dim=1)
            labels = torch.cat([
                labels,
                torch.full((B, 1), -100, device=device, dtype=labels.dtype)
            ], dim=1)

        # 5. Concatenate prefix + text embeddings
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        # 6. Extended attention mask
        prefix_mask = torch.ones(B, prefix_embeds.shape[1], device=device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 7. Combined labels (prefix tokens have label -100 = ignored)
        prefix_labels = torch.full(
            (B, prefix_embeds.shape[1]), -100,
            device=device, dtype=labels.dtype
        )
        combined_labels = torch.cat([prefix_labels, labels], dim=1)

        # 8. Forward through language model (not under no_grad — gradients flow back through inputs_embeds)
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
        )
        hidden_states = lm_out.last_hidden_state  # (B, num_queries+L, lm_hidden)

        # 9. Project to vocab and compute loss
        # Use OPT's lm_head weight (we borrow it, frozen)
        lm_head = self.language_model.get_input_embeddings()
        # OPT ties embedding weights, so get_input_embeddings() gives us the weight
        # But actually for tied embeddings we need to be careful
        # Let me use a separate linear layer that uses the tied weight
        logits = nn.functional.linear(hidden_states, lm_head.weight)  # (B, seq, vocab)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = combined_labels[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return {"loss": loss, "logits": logits}

    def generate(self, pixel_values, opt_tokenizer, max_new_tokens=30,
                 num_beams=1, do_sample=False):
        """
        推理时用：输入图片 → 生成 caption。
        使用 OPTForCausalLM 的 generate 方法。
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # 1. Vision encoder → Q-Former → prefix embeddings
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values)
            image_features = vision_out.last_hidden_state
            image_features = self.vision_proj(image_features)
            qformer_out = self.qformer(image_features)
            prefix_embeds = self.projection(qformer_out)  # (B, num_queries, lm_hidden)

        # 2. Use OPT's generate: pass prefix as input embeddings
        # Start with BOS token
        bos_token_id = opt_tokenizer.bos_token_id or opt_tokenizer.pad_token_id or 0
        input_ids = torch.full((B, 1), bos_token_id, device=device, dtype=torch.long)

        embed_layer = self.language_model.get_input_embeddings()

        generated_ids = []
        for _ in range(max_new_tokens):
            text_embeds = embed_layer(input_ids)  # (B, cur_len, lm_hidden)
            inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

            # attention mask
            prefix_mask = torch.ones(B, prefix_embeds.shape[1], device=device)
            text_mask = torch.ones(B, text_embeds.shape[1], device=device)
            extended_mask = torch.cat([prefix_mask, text_mask], dim=1)

            # Pad if needed
            seq_len = inputs_embeds.shape[1]
            if seq_len % 2 != 0:
                inputs_embeds = torch.cat([
                    inputs_embeds,
                    torch.zeros(B, 1, inputs_embeds.shape[2], device=device, dtype=inputs_embeds.dtype)
                ], dim=1)
                extended_mask = torch.cat([
                    extended_mask,
                    torch.ones(B, 1, device=device)
                ], dim=1)

            lm_out = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_mask,
            )
            hidden = lm_out.last_hidden_state  # (B, seq, lm_hidden)

            # Logits for the last position
            last_hidden = hidden[:, -1, :]  # (B, lm_hidden)
            logits = nn.functional.linear(last_hidden, embed_layer.weight)  # (B, vocab)

            if do_sample:
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            generated_ids.append(next_token)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if all generate EOS
            if (next_token == opt_tokenizer.eos_token_id).all():
                break

        generated_ids = torch.cat(generated_ids, dim=1) if generated_ids else input_ids[:, 1:]
        return generated_ids
