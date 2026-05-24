"""
推理脚本：加载训练好的模型，对图片生成英文 caption，并支持生成可视化 HTML 结果页面。
"""

import os
import sys
import io
import torch
import base64
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MiniBLIP2


def load_model(checkpoint_path, device="cpu"):
    """加载 checkpoint 并重建模型。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = MiniBLIP2(
        vision_model_name=cfg["vision_model"],
        language_model_name=cfg["language_model"],
        num_queries=cfg["num_queries"],
        qformer_hidden=cfg["qformer_hidden"],
        qformer_layers=cfg["qformer_layers"],
        qformer_heads=cfg["qformer_heads"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def generate_caption(model, image, clip_processor, opt_tokenizer,
                     device="cpu", max_new_tokens=32, temperature=0.7):
    """给定一张 PIL Image，生成 caption。temperature=0 为贪心解码。"""
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_out = model.vision_encoder(pixel_values)
        image_features = vision_out.last_hidden_state
        image_features = model.vision_proj(image_features)
        qformer_out = model.qformer(image_features)
        prefix_embeds = model.projection(qformer_out)  # (1, num_queries, lm_hidden)

        embed_layer = model.language_model.get_input_embeddings()

        bos_id = opt_tokenizer.bos_token_id or 0
        input_ids = torch.full((1, 1), bos_id, device=device, dtype=torch.long)
        generated = []

        for _ in range(max_new_tokens):
            text_embeds = embed_layer(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

            pre_mask = torch.ones(1, prefix_embeds.shape[1], device=device)
            txt_mask = torch.ones(1, text_embeds.shape[1], device=device)
            attn_mask = torch.cat([pre_mask, txt_mask], dim=1)

            seq_len = inputs_embeds.shape[1]
            if seq_len % 2 != 0:
                inputs_embeds = torch.cat([
                    inputs_embeds,
                    torch.zeros(1, 1, inputs_embeds.shape[2], device=device, dtype=inputs_embeds.dtype)
                ], dim=1)
                attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=device)], dim=1)

            lm_out = model.language_model(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            hidden = lm_out.last_hidden_state[:, -1, :]
            logits = torch.nn.functional.linear(hidden, embed_layer.weight)

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = next_token.item()

            if token_id == opt_tokenizer.eos_token_id:
                break

            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return opt_tokenizer.decode(generated, skip_special_tokens=True)


def show_examples(model, data_dir, captions_file, clip_processor, opt_tokenizer,
                  num_examples=5, device="cpu"):
    """从验证集中展示几个例子：图片 + 真实 caption + 模型生成 caption。"""
    import pandas as pd

    df = pd.read_csv(captions_file)
    unique_images = df["image"].unique()[:200]
    n = len(unique_images)
    val_images = unique_images[int(n * 0.8):]

    print(f"\n{'='*60}")
    print(f"Showing {num_examples} examples from {len(val_images)} validation images")
    print(f"{'='*60}\n")

    for i, img_name in enumerate(val_images[:num_examples]):
        img_path = os.path.join(data_dir, "images", img_name)
        image = Image.open(img_path).convert("RGB")

        real_captions = df[df["image"] == img_name]["caption"].values
        real_caption = real_captions[0] if len(real_captions) > 0 else "N/A"

        gen_caption = generate_caption(
            model, image, clip_processor, opt_tokenizer,
            device=device, temperature=0.7,
        )

        print(f"Image {i+1}: {img_name}")
        print(f"  Real:     {real_caption}")
        print(f"  Generated: {gen_caption}")
        print()

    return val_images[:num_examples]


def generate_html_results(model, clip_processor, opt_tokenizer, device,
                          data_dir, captions_file, num_images=5, output_path="results.html"):
    """生成可视化 HTML 结果页面：展示图片 + Ground Truth + Greedy + Temperature Sampling。"""
    import pandas as pd

    available = set(os.listdir(os.path.join(data_dir, "images")))
    df = pd.read_csv(captions_file)
    df = df[df["image"].isin(available)]
    unique_images = sorted(df["image"].unique())[:200]
    val_images = unique_images[int(len(unique_images) * 0.8):]

    results = []
    print("Generating captions...")
    for img_name in val_images[:num_images]:
        img_path = os.path.join(data_dir, "images", img_name)
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        image = Image.open(img_path).convert("RGB")
        real_captions = df[df["image"] == img_name]["caption"].values.tolist()

        greedy = generate_caption(model, image, clip_processor, opt_tokenizer,
                                  device=device, temperature=0)
        sample = generate_caption(model, image, clip_processor, opt_tokenizer,
                                  device=device, temperature=0.7)

        results.append({
            "name": img_name,
            "b64": img_b64,
            "real": real_captions,
            "greedy": greedy,
            "sample": sample,
        })
        print(f"  {img_name} done.")

    # Build HTML
    html = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Mini-BLIP2 推理结果</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f5f5; padding: 30px; }
h1 { text-align: center; color: #333; margin-bottom: 8px; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }
.card { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 28px; overflow: hidden; max-width: 960px; margin-left: auto; margin-right: auto; }
.card-header { background: #1a1a2e; color: #fff; padding: 12px 20px; font-size: 15px; }
.card-body { display: flex; gap: 20px; padding: 20px; }
.card-image { flex: 0 0 320px; }
.card-image img { width: 100%; border-radius: 8px; border: 1px solid #eee; }
.card-text { flex: 1; min-width: 0; }
.section { margin-bottom: 14px; }
.section-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.label-real { color: #2e7d32; }
.label-gen { color: #1565c0; }
.label-sample { color: #6a1b9a; }
.text-real { background: #e8f5e9; color: #2e7d32; padding: 8px 12px; border-radius: 6px; font-size: 14px; line-height: 1.5; margin-bottom: 3px; }
.text-gen { background: #e3f2fd; color: #0d47a1; padding: 8px 12px; border-radius: 6px; font-size: 14px; line-height: 1.5; }
.text-sample { background: #f3e5f5; color: #4a148c; padding: 8px 12px; border-radius: 6px; font-size: 14px; line-height: 1.5; }
.summary { max-width: 960px; margin: 30px auto; padding: 20px; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.summary h3 { color: #333; margin-bottom: 12px; }
.summary table { width: 100%; border-collapse: collapse; font-size: 14px; }
.summary th { background: #f5f5f5; padding: 8px 12px; text-align: left; border-bottom: 2px solid #ddd; }
.summary td { padding: 8px 12px; border-bottom: 1px solid #eee; }
</style>
</head>
<body>
<h1>Mini-BLIP2 模型推理结果</h1>
<p class="subtitle">Best Checkpoint | Q-Former / OPT-125M</p>
"""

    for i, r in enumerate(results):
        real_html = "".join(f'<div class="text-real">[{j+1}] {c}</div>' for j, c in enumerate(r["real"]))
        html += f"""
<div class="card">
  <div class="card-header">Image {i+1}: {r['name']}</div>
  <div class="card-body">
    <div class="card-image">
      <img src="data:image/jpeg;base64,{r['b64']}" alt="{r['name']}" />
    </div>
    <div class="card-text">
      <div class="section">
        <div class="section-label label-real">Ground Truth</div>
        {real_html}
      </div>
      <div class="section">
        <div class="section-label label-gen">Greedy Decoding</div>
        <div class="text-gen">{r['greedy']}</div>
      </div>
      <div class="section">
        <div class="section-label label-sample">Temperature Sampling (t=0.7)</div>
        <div class="text-sample">{r['sample']}</div>
      </div>
    </div>
  </div>
</div>"""

    html += """
<div class="summary">
<h3>Training Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Training Data</td><td>160 images x 5 captions = 800 samples</td></tr>
<tr><td>Validation Data</td><td>40 images x 5 captions = 200 samples</td></tr>
</table>
</div>
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDone! Open: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_checkpoint.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--captions_file", type=str, default="data/captions.txt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for captioning")
    parser.add_argument("--html", type=str, nargs="?", const="results.html", default=None,
                        help="Generate HTML results page (optional: output path)")
    args = parser.parse_args()

    # Path handling
    base = os.path.join(os.path.dirname(__file__), "..")
    checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(base, args.checkpoint)
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(base, args.data_dir)
    captions_file = args.captions_file if os.path.isabs(args.captions_file) else os.path.join(base, args.captions_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load
    model, cfg = load_model(checkpoint_path, device)
    clip_processor = CLIPProcessor.from_pretrained(cfg["vision_model"])
    opt_tokenizer = AutoTokenizer.from_pretrained(cfg["language_model"])

    if args.html is not None:
        output_path = args.html if args.html != "results.html" else os.path.join(base, args.html)
        generate_html_results(model, clip_processor, opt_tokenizer, device,
                              data_dir, captions_file,
                              num_images=args.num_examples, output_path=output_path)
    elif args.image:
        image = Image.open(args.image).convert("RGB")
        caption = generate_caption(model, image, clip_processor, opt_tokenizer, device=device)
        print(f"Generated: {caption}")
    else:
        show_examples(model, data_dir, captions_file, clip_processor, opt_tokenizer,
                      num_examples=args.num_examples, device=device)
