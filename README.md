# ComfyUI Multi Prompt Generator

[English](#english) | [한국어](#한국어)

---

## English

A ComfyUI custom node that processes a list of prompts in a single run to generate multiple images.
Includes upscaling, LUT application, and real-time preview.

### Features

- **Batch Generation** — Process line-separated prompt list at once
- **2-Pass Upscale** — Upscale model + img2img refinement
- **Size Alignment** — 64/8 pixel alignment to prevent edge artifacts
- **LUT Support** — Apply .cube LUT files
- **Real-time Preview** — Check results during generation, cancel if needed
- **Auto Save** — Automatic filename based on first tag

### Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mrm987/ComfyUI_Multi_Prompt_Generator.git
```
Restart ComfyUI.

### Usage

1. Search node: **"Multi Prompt Generator"**
2. Connect: Model, CLIP, VAE, Empty Latent, Upscale Model
3. Enter prompt list — one per line:
```
smile, happy, bright eyes
angry, furrowed brow
sad, downcast eyes
crying, tears
```

### Inputs

#### Required

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | Checkpoint model |
| clip | CLIP | CLIP model |
| vae | VAE | VAE model |
| latent | LATENT | Empty Latent Image |
| upscale_model | UPSCALE_MODEL | Upscale model (e.g., 2x-AnimeSharp) |
| base_prompt | STRING | Base prompt (combined with each line) |
| negative_prompt | STRING | Negative prompt |
| prompt_list | STRING | Prompt list (line-separated) |
| seed | INT | Seed |
| steps | INT | Sampling steps (default: 30) |
| cfg | FLOAT | CFG scale (default: 5.0) |
| enable_upscale | BOOL | Enable upscale (default: True) |
| save_prefix | STRING | Output folder name (default: "MultiPrompt") |

#### Optional

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| scale_factor | FLOAT | 0.7 | Downscale ratio after upscale |
| upscale_steps | INT | 15 | 2nd pass steps |
| upscale_cfg | FLOAT | 5.0 | 2nd pass CFG |
| upscale_denoise | FLOAT | 0.5 | 2nd pass denoise |
| size_alignment | COMBO | "64" | Size alignment (64 / 8 / none) |
| lut_name | COMBO | "None" | LUT file selection |
| lut_strength | FLOAT | 0.3 | LUT strength |
| enable_preview | BOOL | True | Real-time preview |

### Output

```
output/[save_prefix]/01_smile_00001.png
output/[save_prefix]/02_angry_00001.png
output/[save_prefix]/03_sad_00001.png
...
```

Filename format: `[index]_[first_tag]_[counter].png`

### Pipeline

```
For each prompt:
  1. Combine base_prompt + current line
  2. KSampler (1st pass)
  3. Show preview
  4. Upscale Model → Size alignment → KSampler (2nd pass)
  5. Apply LUT
  6. Update preview
  7. Save
```

### Notes

- **size_alignment**: 64 is safest. Use 64 if you see white edges
- **LUT files**: Auto-detected from `comfyui-propost/LUTs/` or `models/luts/` folder
- **Preview**: Check 1st pass result, cancel with Cancel button if not satisfied

---

## 한국어

프롬프트 리스트를 한 번에 처리해서 여러 이미지를 생성하는 ComfyUI 커스텀 노드.
업스케일, LUT 적용, 실시간 프리뷰 기능 포함.

### 기능

- **다중 프롬프트 배치 생성** — 줄바꿈으로 구분된 프롬프트 리스트를 한 번에 처리
- **2-Pass 업스케일** — 업스케일 모델 + img2img 리파인
- **크기 정렬** — 64/8배수 정렬로 테두리 아티팩트 방지
- **LUT 적용** — .cube LUT 파일 지원
- **실시간 프리뷰** — 생성 중 결과 확인, 마음에 안 들면 취소 가능
- **자동 저장** — 첫 태그 기반 파일명 자동 생성

### 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mrm987/ComfyUI_Multi_Prompt_Generator.git
```
ComfyUI 재시작.

### 사용법

1. 노드 검색: **"Multi Prompt Generator"**
2. 연결: Model, CLIP, VAE, Empty Latent, Upscale Model
3. 프롬프트 리스트 입력 — 한 줄에 하나씩:
```
smile, happy, bright eyes
angry, furrowed brow
sad, downcast eyes
crying, tears
```

### 입력

#### 필수

| Input | Type | Description |
|-------|------|-------------|
| model | MODEL | 체크포인트 모델 |
| clip | CLIP | CLIP 모델 |
| vae | VAE | VAE 모델 |
| latent | LATENT | Empty Latent Image |
| upscale_model | UPSCALE_MODEL | 업스케일 모델 (예: 2x-AnimeSharp) |
| base_prompt | STRING | 기본 프롬프트 (각 줄과 결합됨) |
| negative_prompt | STRING | 네거티브 프롬프트 |
| prompt_list | STRING | 프롬프트 리스트 (줄바꿈 구분) |
| seed | INT | 시드 |
| steps | INT | 샘플링 스텝 (기본: 30) |
| cfg | FLOAT | CFG 스케일 (기본: 5.0) |
| enable_upscale | BOOL | 업스케일 활성화 (기본: True) |
| save_prefix | STRING | 저장 폴더명 (기본: "MultiPrompt") |

#### 선택

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| scale_factor | FLOAT | 0.7 | 업스케일 후 다운스케일 비율 |
| upscale_steps | INT | 15 | 2차 샘플링 스텝 |
| upscale_cfg | FLOAT | 5.0 | 2차 CFG |
| upscale_denoise | FLOAT | 0.5 | 2차 디노이즈 |
| size_alignment | COMBO | "64" | 크기 정렬 (64 / 8 / none) |
| lut_name | COMBO | "None" | LUT 파일 선택 |
| lut_strength | FLOAT | 0.3 | LUT 강도 |
| enable_preview | BOOL | True | 실시간 프리뷰 |

### 출력

```
output/[save_prefix]/01_smile_00001.png
output/[save_prefix]/02_angry_00001.png
output/[save_prefix]/03_sad_00001.png
...
```

파일명: `[순번]_[첫태그]_[카운터].png`

### 파이프라인

```
각 프롬프트마다:
  1. base_prompt + 현재 줄 결합
  2. KSampler (1차)
  3. 프리뷰 표시
  4. Upscale Model → 크기 정렬 → KSampler (2차)
  5. LUT 적용
  6. 프리뷰 업데이트
  7. 저장
```

### 참고

- **size_alignment**: 64가 가장 안전. 흰 테두리 생기면 64 사용
- **LUT 파일**: `comfyui-propost/LUTs/` 또는 `models/luts/` 폴더의 .cube 파일 자동 인식
- **프리뷰**: 1차 결과 확인 후 마음에 안 들면 Cancel로 중단 가능

---

## License

MIT
