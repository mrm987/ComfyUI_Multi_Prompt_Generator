# ComfyUI Multi Prompt Generator

Generate multiple images from a prompt list in a single run, with built-in upscaling and LUT support.

![preview](https://github.com/mrm987/ComfyUI_Multi_Prompt_Generator/assets/preview.png)

## Features

- **Batch Generation** — Input a list of prompts (one per line), generate all at once
- **Auto Upscaling** — 2-pass upscale with model upscaler + img2img refinement
- **64x Alignment** — Automatically adjusts dimensions to prevent edge artifacts
- **LUT Support** — Apply .cube LUT files with adjustable strength
- **Live Preview** — See results during generation (toggleable)
- **Auto Save** — Images saved with auto-numbered filenames based on first tag

## Installation

### Option 1: ComfyUI Manager (Coming Soon)
Search for "Multi Prompt Generator" in ComfyUI Manager.

### Option 2: Manual Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mrm987/ComfyUI_Multi_Prompt_Generator.git
```
Restart ComfyUI.

## Usage

1. Search for **"Multi Prompt Generator"** in the node menu
2. Connect: Model, CLIP, VAE, Empty Latent, (optional) Upscale Model
3. Set your base prompt and negative prompt
4. Enter prompt list — one variation per line:
```
smile, happy, bright eyes
angry, furrowed brow, clenched teeth
sad, downcast eyes, closed mouth
crying, tears, watery eyes
```

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Checkpoint model |
| clip | CLIP | CLIP model |
| vae | VAE | VAE model |
| latent | LATENT | Empty latent image |
| base_prompt | STRING | Base prompt (combined with each line) |
| negative_prompt | STRING | Negative prompt |
| prompt_list | STRING | One prompt variation per line |
| seed | INT | Random seed |
| steps | INT | Sampling steps (default: 30) |
| cfg | FLOAT | CFG scale (default: 5.0) |
| enable_upscale | BOOL | Enable 2-pass upscaling (default: True) |
| enable_preview | BOOL | Show preview during generation (default: True) |
| save_prefix | STRING | Output folder name (default: "MultiPrompt") |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| upscale_model | UPSCALE_MODEL | Upscale model (e.g., 2x-AnimeSharp) |
| scale_factor | FLOAT | Downscale after upscale (default: 0.7) |
| upscale_steps | INT | 2nd pass steps (default: 15) |
| upscale_cfg | FLOAT | 2nd pass CFG (default: 5.0) |
| upscale_denoise | FLOAT | 2nd pass denoise (default: 0.5) |
| lut_name | COMBO | LUT file from comfyui-propost/LUTs |
| lut_strength | FLOAT | LUT intensity (default: 0.3) |

## Output

Images are saved to:
```
output/[save_prefix]/01_smile_00001.png
output/[save_prefix]/02_angry_00001.png
output/[save_prefix]/03_sad_00001.png
...
```

Filename format: `[index]_[first_tag]_[counter].png`

## Pipeline

```
For each prompt line:
  1. Combine: base_prompt + line
  2. KSampler (1st pass)
  3. Preview (if enabled)
  4. Upscale Model → Resize to 64x → KSampler (2nd pass)
  5. Apply LUT (if selected)
  6. Preview update (if enabled)
  7. Save
```

## Requirements

- ComfyUI
- (Optional) [comfyui-propost](https://github.com/Beinsezii/comfyui-propost) for LUT files

## License

MIT
