import os
import sys
import subprocess
import importlib
import glob
import io
import torch
import numpy as np
from PIL import Image
import folder_paths

import comfy.sd
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management

from server import PromptServer


def ensure_package(package, install_package_name=None):
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"[MultiPrompt] Package {package} not found. Installing...")
        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            pip_install = [sys.executable, "-s", "-m", "pip", "install"]
        else:
            pip_install = [sys.executable, "-m", "pip", "install"]
        subprocess.check_call(pip_install + [install_package_name or package])

ensure_package("dotenv", "python-dotenv")
ensure_package("requests")


def get_lut_files():
    """LUT 파일 목록 가져오기"""
    lut_paths = []
    
    # comfyui-propost LUTs 폴더
    propost_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0] if folder_paths.get_folder_paths("custom_nodes") else "", "comfyui-propost", "LUTs")
    if os.path.exists(propost_path):
        lut_paths.append(propost_path)
    
    # models/luts 폴더
    models_lut_path = os.path.join(folder_paths.models_dir, "luts")
    if os.path.exists(models_lut_path):
        lut_paths.append(models_lut_path)
    
    # custom_nodes 폴더 내 LUTs
    custom_nodes_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    propost_luts = os.path.join(custom_nodes_path, "comfyui-propost", "LUTs")
    if os.path.exists(propost_luts) and propost_luts not in lut_paths:
        lut_paths.append(propost_luts)
    
    lut_files = ["None"]
    for path in lut_paths:
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.lower().endswith(".cube"):
                    lut_files.append(f)
    
    return list(set(lut_files))


def find_lut_file(lut_name):
    """LUT 파일 전체 경로 찾기"""
    if lut_name == "None":
        return None
    
    search_paths = []
    
    custom_nodes_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_paths.append(os.path.join(custom_nodes_path, "comfyui-propost", "LUTs"))
    search_paths.append(os.path.join(folder_paths.models_dir, "luts"))
    
    for path in search_paths:
        full_path = os.path.join(path, lut_name)
        if os.path.exists(full_path):
            return full_path
    
    return None


def parse_cube_lut(lut_path):
    """Parse .cube LUT file"""
    with open(lut_path, 'r') as f:
        lines = f.readlines()
    
    lut_size = 0
    lut_data = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        if line.startswith('TITLE'):
            continue
        if line.startswith('LUT_3D_SIZE'):
            lut_size = int(line.split()[1])
            continue
        if line.startswith('DOMAIN_MIN') or line.startswith('DOMAIN_MAX'):
            continue
        
        try:
            values = [float(x) for x in line.split()]
            if len(values) == 3:
                lut_data.append(values)
        except:
            continue
    
    if lut_size == 0 or len(lut_data) == 0:
        return None, 0
    
    # cube 파일은 R이 가장 빠르게 변함 -> Fortran order로 reshape
    lut_array = np.array(lut_data, dtype=np.float32)
    lut_3d = np.reshape(lut_array, (lut_size, lut_size, lut_size, 3), order='F')
    
    return lut_3d, lut_size


def apply_lut(image, lut_3d, lut_size, strength=1.0):
    """Apply 3D LUT to image using trilinear interpolation"""
    if lut_3d is None or strength <= 0:
        return image
    
    # image: [B, H, W, C] tensor, 0-1 range
    img_np = image.cpu().to(torch.float32).numpy()
    result = []
    
    for b in range(img_np.shape[0]):
        img = img_np[b][:, :, :3].copy()
        img = np.clip(img, 0, 1)
        
        # 이미지를 LUT 인덱스로 스케일
        h, w, _ = img.shape
        img_flat = img.reshape(-1, 3)
        
        # 스케일링
        coords = img_flat * (lut_size - 1)
        
        # 정수 인덱스와 소수 부분
        coords_floor = np.floor(coords).astype(np.int32)
        coords_ceil = np.minimum(coords_floor + 1, lut_size - 1)
        frac = coords - coords_floor
        
        # 인덱스 클램프
        coords_floor = np.clip(coords_floor, 0, lut_size - 1)
        
        r0, g0, b0 = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
        r1, g1, b1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]
        fr, fg, fb = frac[:, 0], frac[:, 1], frac[:, 2]
        
        # 8개 꼭짓점 값
        c000 = lut_3d[r0, g0, b0]
        c001 = lut_3d[r0, g0, b1]
        c010 = lut_3d[r0, g1, b0]
        c011 = lut_3d[r0, g1, b1]
        c100 = lut_3d[r1, g0, b0]
        c101 = lut_3d[r1, g0, b1]
        c110 = lut_3d[r1, g1, b0]
        c111 = lut_3d[r1, g1, b1]
        
        # 8개 가중치 (colour-science 방식)
        w000 = ((1 - fr) * (1 - fg) * (1 - fb))[:, None]
        w001 = ((1 - fr) * (1 - fg) * fb)[:, None]
        w010 = ((1 - fr) * fg * (1 - fb))[:, None]
        w011 = ((1 - fr) * fg * fb)[:, None]
        w100 = (fr * (1 - fg) * (1 - fb))[:, None]
        w101 = (fr * (1 - fg) * fb)[:, None]
        w110 = (fr * fg * (1 - fb))[:, None]
        w111 = (fr * fg * fb)[:, None]
        
        # trilinear interpolation
        lut_result = (c000 * w000 + c001 * w001 + c010 * w010 + c011 * w011 +
                      c100 * w100 + c101 * w101 + c110 * w110 + c111 * w111)
        
        lut_result = lut_result.reshape(h, w, 3)
        
        # strength 블렌딩
        if strength < 1.0:
            lut_result = img * (1 - strength) + lut_result * strength
        
        lut_result = np.clip(lut_result, 0, 1).astype(np.float32)
        result.append(lut_result)
    
    result = np.stack(result, axis=0)
    return torch.from_numpy(result).to(image.device)


class MultiPromptGenerator:
    """
    프롬프트 리스트를 입력받아 한 번에 여러 이미지를 생성하고 업스케일하는 노드
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "upscale_model": ("UPSCALE_MODEL",),
                "base_prompt": ("STRING", {"multiline": True, "default": "1girl, solo,"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad quality,"}),
                "prompt_list": ("STRING", {"multiline": True, "default": "# Blank line = new prompt\n# 빈 줄 = 새 프롬프트\n# Use skip_indices to skip (e.g. 3,4,7)\n\nsmile, happy,\nbright eyes\n\nangry, furrowed brow\n\nsad, crying, tears"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "enable_upscale": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "MultiPrompt"}),
            },
            "optional": {
                "skip_indices": ("STRING", {"default": "", "placeholder": "e.g. 3,4,7"}),
                "downscale_ratio": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01}),
                "upscale_steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "upscale_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "upscale_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "size_alignment": (["none", "8", "64"], {"default": "none"}),
                "lut_name": (get_lut_files(),),
                "lut_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    def encode_prompt(self, clip, text):
        """CLIP 텍스트 인코딩"""
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def do_sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        """샘플링 실행"""
        device = comfy.model_management.get_torch_device()
        latent = latent_image.clone().to(device)
        
        noise = comfy.sample.prepare_noise(latent, seed, None)
        
        samples = comfy.sample.sample(
            model, 
            noise, 
            steps, 
            cfg, 
            sampler_name,
            scheduler,
            positive, 
            negative, 
            latent,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=True,
            noise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed
        )
        
        return samples

    def decode_vae(self, vae, samples):
        """VAE 디코딩"""
        return vae.decode(samples)

    def encode_vae(self, vae, image):
        """VAE 인코딩"""
        return vae.encode(image[:, :, :, :3])

    def upscale_with_model(self, upscale_model, image):
        """업스케일 모델로 이미지 확대"""
        device = comfy.model_management.get_torch_device()
        
        upscale_model.to(device)
        in_img = image.movedim(-1, -3).to(device)
        
        tile = 512
        overlap = 32
        
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar
                )
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
        
        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        return s

    def resize_with_alignment(self, image, scale_factor, alignment="64"):
        """이미지를 지정된 배수 크기로 리사이즈"""
        batch, height, width, channels = image.shape
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 배수 정렬
        if alignment == "64":
            new_width = ((new_width + 32) // 64) * 64
            new_height = ((new_height + 32) // 64) * 64
            new_width = max(64, new_width)
            new_height = max(64, new_height)
        elif alignment == "8":
            new_width = ((new_width + 4) // 8) * 8
            new_height = ((new_height + 4) // 8) * 8
            new_width = max(8, new_width)
            new_height = max(8, new_height)
        # "none"이면 그대로
        
        # BCHW로 변환 후 리사이즈
        samples = image.movedim(-1, 1)
        s = comfy.utils.common_upscale(samples, new_width, new_height, "lanczos", "disabled")
        s = s.movedim(1, -1)
        
        return s

    def save_image(self, image, prefix, filename, counter, prompt=None, extra_pnginfo=None):
        """이미지 저장 (메타데이터 포함)"""
        import json
        from PIL.PngImagePlugin import PngInfo
        
        full_output_folder = os.path.join(self.output_dir, prefix)
        os.makedirs(full_output_folder, exist_ok=True)
        
        file = f"{filename}_{counter:05d}.png"
        filepath = os.path.join(full_output_folder, file)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # 메타데이터 추가
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        
        pil_img.save(filepath, pnginfo=metadata, compress_level=4)
        
        print(f"[MultiPrompt] Saved: {filepath}")
        return filepath

    def send_preview(self, image, unique_id, idx, stage=""):
        """생성 중 프리뷰를 UI로 전송"""
        import time
        temp_dir = folder_paths.get_temp_directory()
        
        # temp 폴더에 프리뷰 저장 (타임스탬프로 캐시 우회)
        timestamp = int(time.time() * 1000)
        preview_filename = f"multiprompt_preview_{unique_id}_{idx}_{stage}_{timestamp}.png"
        preview_path = os.path.join(temp_dir, preview_filename)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(preview_path, compress_level=1)
        
        # UI로 프리뷰 전송
        PromptServer.instance.send_sync("executed", {
            "node": unique_id,
            "output": {
                "images": [{
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp"
                }]
            }
        })

    def generate(self, model, clip, vae, latent, upscale_model, base_prompt, negative_prompt, prompt_list,
                 seed, steps, cfg, sampler_name, scheduler, enable_upscale, save_prefix,
                 skip_indices="", downscale_ratio=0.7, upscale_steps=15, 
                 upscale_cfg=5.0, upscale_denoise=0.5, size_alignment="none",
                 lut_name="None", lut_strength=0.3, enable_preview=True, 
                 unique_id=None, prompt=None, extra_pnginfo=None):
        
        # prompt_list 파싱 (빈 줄로 구분, # 주석 무시, - 시작하면 블록 스킵)
        blocks = prompt_list.strip().split("\n\n")
        lines = []
        for block in blocks:
            block_lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if not block_lines:
                continue
            
            # 첫 번째 유효 줄(주석 아닌)이 -로 시작하면 블록 스킵
            first_content = None
            for line in block_lines:
                if not line.startswith("#"):
                    first_content = line
                    break
            
            if first_content and first_content.startswith("-"):
                continue  # 블록 스킵
            
            # 블록 내 줄바꿈을 공백으로 합침, # 주석 제외
            merged = " ".join(
                line.strip() for line in block_lines 
                if not line.startswith("#")
            )
            if merged:
                lines.append(merged)
        
        # skip_indices 파싱 (1-based)
        skip_set = set()
        if skip_indices.strip():
            for part in skip_indices.split(","):
                part = part.strip()
                if part.isdigit():
                    skip_set.add(int(part))
        
        if not lines:
            raise ValueError("prompt_list가 비어있습니다.")
        
        # LUT 로드
        lut_data = None
        lut_size = 0
        if lut_name != "None":
            lut_path = find_lut_file(lut_name)
            if lut_path:
                lut_data, lut_size = parse_cube_lut(lut_path)
                print(f"[MultiPrompt] LUT loaded: {lut_name} (size: {lut_size})")
        
        # negative 인코딩 (한 번만)
        negative_cond = self.encode_prompt(clip, negative_prompt)
        
        all_images = []
        counter = self._get_next_counter(save_prefix)
        
        # 인터럽트 체크용
        import comfy.model_management as mm
        
        for idx, line in enumerate(lines):
            # 중단 체크
            mm.throw_exception_if_processing_interrupted()
            
            # skip_indices에 있으면 스킵 (1-based)
            if (idx + 1) in skip_set:
                print(f"[MultiPrompt] Skipping {idx + 1}/{len(lines)}: {line[:40]}...")
                continue
            
            print(f"[MultiPrompt] Processing {idx + 1}/{len(lines)}: {line[:40]}...")
            
            # 파일명: 순번 + 첫 태그
            first_tag = line.split(",")[0].strip().replace(" ", "_")
            filename = f"{idx + 1:02d}_{first_tag}"
            
            # 프롬프트 결합
            full_prompt = f"{base_prompt}, {line}"
            positive_cond = self.encode_prompt(clip, full_prompt)
            
            # latent 복사
            latent_samples = latent["samples"].clone()
            
            # 1단계: 기본 샘플링
            samples = self.do_sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond,
                latent_samples, denoise=1.0
            )
            image = self.decode_vae(vae, samples)
            
            # 1차 결과 프리뷰 전송
            if enable_preview and unique_id is not None:
                self.send_preview(image, unique_id, idx, "1st")
            
            # 2단계: 업스케일
            if enable_upscale and upscale_model is not None:
                # 업스케일 모델 적용
                upscaled = self.upscale_with_model(upscale_model, image)
                
                # 64배수로 리사이즈
                resized = self.resize_with_alignment(upscaled, downscale_ratio, size_alignment)
                
                # VAE 인코딩 → 2차 샘플링 → 디코딩
                latent_up = self.encode_vae(vae, resized)
                
                samples_up = self.do_sample(
                    model, seed, upscale_steps, upscale_cfg, sampler_name, scheduler,
                    positive_cond, negative_cond,
                    latent_up, denoise=upscale_denoise
                )
                image = self.decode_vae(vae, samples_up)
            
            # 3단계: LUT 적용
            if lut_data is not None and lut_strength > 0:
                image = apply_lut(image, lut_data, lut_size, lut_strength)
            
            # 최종 결과 프리뷰 교체
            if enable_preview and unique_id is not None:
                self.send_preview(image, unique_id, idx, "final")
            
            # 저장
            self.save_image(image, save_prefix, filename, counter, prompt, extra_pnginfo)
            all_images.append(image)
        
        print(f"[MultiPrompt] Complete! Generated {len(all_images)} images.")
        
        # 마지막 이미지를 UI에 표시
        if all_images and enable_preview:
            last_image = all_images[-1]
            import time
            temp_dir = folder_paths.get_temp_directory()
            timestamp = int(time.time() * 1000)
            preview_filename = f"multiprompt_final_{timestamp}.png"
            preview_path = os.path.join(temp_dir, preview_filename)
            
            img = last_image[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_img.save(preview_path, compress_level=1)
            
            return {"ui": {"images": [{"filename": preview_filename, "subfolder": "", "type": "temp"}]}, "result": (all_images,)}
        
        return {"ui": {"images": []}, "result": (all_images,)}

    def _get_next_counter(self, prefix):
        """다음 카운터 번호"""
        full_output_folder = os.path.join(self.output_dir, prefix)
        
        if not os.path.exists(full_output_folder):
            return 1
        
        existing_files = os.listdir(full_output_folder)
        if not existing_files:
            return 1
        
        max_counter = 0
        for f in existing_files:
            if f.endswith(".png"):
                try:
                    counter_str = f.rsplit("_", 1)[-1].replace(".png", "")
                    counter_val = int(counter_str)
                    max_counter = max(max_counter, counter_val)
                except:
                    pass
        
        return max_counter + 1


def load_nai_token():
    """ComfyUI 루트의 .env에서 NAI_ACCESS_TOKEN 로드"""
    from dotenv import dotenv_values
    
    # ComfyUI 루트 경로들
    possible_paths = [
        os.path.join(folder_paths.base_path, ".env") if hasattr(folder_paths, 'base_path') else None,
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
    ]
    
    for env_path in possible_paths:
        if env_path and os.path.exists(env_path):
            env = dotenv_values(env_path)
            if "NAI_ACCESS_TOKEN" in env:
                return env["NAI_ACCESS_TOKEN"]
    
    # 환경변수에서도 시도
    return os.environ.get("NAI_ACCESS_TOKEN", None)


# Character Reference 헬퍼 함수들
ACCEPTED_CR_SIZES = [(1024, 1536), (1536, 1024), (1472, 1472)]

def _choose_cr_canvas(w, h):
    """캐릭터 레퍼런스용 캔버스 크기 선택 (원본 비율에 가장 가까운 것)"""
    aspect = w / h
    best = None
    best_diff = 9e9
    for cw, ch in ACCEPTED_CR_SIZES:
        diff = abs((cw / ch) - aspect)
        if diff < best_diff:
            best_diff = diff
            best = (cw, ch)
    return best

def pad_image_to_canvas(tensor_image, target_size):
    """이미지를 캔버스 크기에 맞게 letterbox 패딩"""
    _, H, W, C = tensor_image.shape
    tw, th = target_size
    arr = (tensor_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if (C == 4) else "RGB"
    pil = Image.fromarray(arr)
    
    scale = min(tw / W, th / H)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    pil_resized = pil.resize((new_w, new_h), Image.LANCZOS)
    
    if mode == "RGBA":
        canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(pil_resized, offset)
    
    out = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out)[None,]

def image_to_base64(tensor_image):
    """텐서 이미지를 base64로 변환"""
    import base64
    arr = (tensor_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class NAIMultiPromptGenerator:
    """
    NovelAI API를 사용하여 다중 프롬프트 이미지 생성 + 업스케일 + LUT
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.api_url = "https://image.novelai.net/ai/generate-image"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_prompt": ("STRING", {"multiline": True, "default": "1girl, solo, best quality, amazing quality, very aesthetic, absurdres,"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad quality,"}),
                "prompt_list": ("STRING", {"multiline": True, "default": "# Blank line = new prompt\n# 빈 줄 = 새 프롬프트\n\nsmile, happy\n\nangry, furrowed brow\n\nsad, crying"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1216, "min": 64, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"],),
                "scheduler": (["native", "karras", "exponential", "polyexponential"],),
                "smea": (["none", "SMEA", "SMEA+DYN"],),
                "nai_model": (["nai-diffusion-4-5-full", "nai-diffusion-4-curated-preview", "nai-diffusion-3"],),
                "save_prefix": ("STRING", {"default": "NAI_MultiPrompt"}),
            },
            "optional": {
                "skip_indices": ("STRING", {"default": "", "placeholder": "e.g. 3,4,7"}),
                "variety": ("BOOLEAN", {"default": False}),
                "decrisper": ("BOOLEAN", {"default": False}),
                "free_only": ("BOOLEAN", {"default": True}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "uncond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01}),
                "reference_image": ("IMAGE",),
                "char_ref_style_aware": ("BOOLEAN", {"default": True, "tooltip": "Copy style along with identity"}),
                "char_ref_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strictly to match the character"}),
                "lut_name": (get_lut_files(),),
                "lut_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True

    def call_nai_api(self, prompt, negative_prompt, width, height, seed, steps, cfg, sampler, scheduler, smea, nai_model, variety, decrisper, cfg_rescale, uncond_scale, reference_image=None, char_ref_style_aware=True, char_ref_fidelity=1.0):
        """NAI API 호출"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import zipfile
        import io
        
        token = load_nai_token()
        if not token:
            raise ValueError("NAI_ACCESS_TOKEN not found. Please set it in ComfyUI/.env file")
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        # SMEA 설정 (ddim에서는 비활성화)
        sm = (smea in ["SMEA", "SMEA+DYN"]) and sampler != "ddim"
        sm_dyn = (smea == "SMEA+DYN") and sampler != "ddim"
        
        # ddim → ddim_v3 (v2 모델 제외)
        actual_sampler = sampler
        if sampler == "ddim" and "nai-diffusion-2" not in nai_model:
            actual_sampler = "ddim_v3"
        
        params = {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": cfg,
            "sampler": actual_sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": 1,
            "ucPreset": 3,
            "qualityToggle": False,
            "sm": sm,
            "sm_dyn": sm_dyn,
            "dynamic_thresholding": decrisper,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
            "legacy_v3_extend": False,
            "uncond_scale": uncond_scale,
            "negative_prompt": negative_prompt,
            "prompt": prompt,
            "reference_image_multiple": [],
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": [],
            "extra_noise_seed": seed,
            "v4_prompt": {
                "use_coords": False,
                "use_order": False,
                "caption": {"base_caption": prompt, "char_captions": []}
            },
            "v4_negative_prompt": {
                "use_coords": False,
                "use_order": False,
                "caption": {"base_caption": negative_prompt, "char_captions": []}
            }
        }
        
        # Character Reference 처리
        if reference_image is not None:
            _, h_raw, w_raw, _ = reference_image.shape
            canvas_w, canvas_h = _choose_cr_canvas(w_raw, h_raw)
            padded = pad_image_to_canvas(reference_image, (canvas_w, canvas_h))
            base_caption = "character&style" if char_ref_style_aware else "character"
            
            params["director_reference_images"] = [image_to_base64(padded)]
            params["director_reference_descriptions"] = [{
                "use_coords": False, 
                "use_order": False, 
                "legacy_uc": False, 
                "caption": {"base_caption": base_caption, "char_captions": []}
            }]
            params["director_reference_strength_values"] = [1.0]
            params["director_reference_secondary_strength_values"] = [1.0 - char_ref_fidelity]
            params["director_reference_information_extracted"] = [1.0]
        
        # k_euler_ancestral + non-native scheduler
        if sampler == "k_euler_ancestral" and scheduler != "native":
            params["deliberate_euler_ancestral_bug"] = False
            params["prefer_brownian"] = True
        
        # variety → skip_cfg_above_sigma 계산
        if variety:
            # NAI 공식: skip_cfg_above_sigma 계산
            params["skip_cfg_above_sigma"] = self._calculate_skip_cfg_above_sigma(width, height)
        
        payload = {
            "input": prompt,
            "model": nai_model,
            "action": "generate",
            "parameters": params
        }
        
        # Retry 로직
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        
        response = session.post(self.api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            raise ValueError(f"NAI API error: {response.status_code} - {response.text}")
        
        # ZIP에서 이미지 추출
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for name in z.namelist():
                if name.endswith('.png'):
                    img_data = z.read(name)
                    pil_img = Image.open(io.BytesIO(img_data))
                    # RGB 변환 후 tensor로
                    img_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
                    return torch.from_numpy(img_np).unsqueeze(0)
        
        raise ValueError("No image found in NAI API response")
    
    def _calculate_skip_cfg_above_sigma(self, width, height):
        """variety용 skip_cfg_above_sigma 계산 (NAI 공식)"""
        # 기본 해상도 기준 스케일링
        base_res = 1024 * 1024
        current_res = width * height
        scale = (current_res / base_res) ** 0.5
        return 19.0 * scale

    def save_image(self, image, prefix, filename, counter, prompt=None, extra_pnginfo=None):
        import json
        from PIL.PngImagePlugin import PngInfo
        
        full_output_folder = os.path.join(self.output_dir, prefix)
        os.makedirs(full_output_folder, exist_ok=True)
        
        file = f"{filename}_{counter:05d}.png"
        filepath = os.path.join(full_output_folder, file)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))
        
        pil_img.save(filepath, pnginfo=metadata, compress_level=4)
        print(f"[NAI MultiPrompt] Saved: {filepath}")
        return filepath

    def send_preview(self, image, unique_id, idx, stage=""):
        import time
        temp_dir = folder_paths.get_temp_directory()
        timestamp = int(time.time() * 1000)
        preview_filename = f"nai_multiprompt_preview_{unique_id}_{idx}_{stage}_{timestamp}.png"
        preview_path = os.path.join(temp_dir, preview_filename)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(preview_path, compress_level=1)
        
        PromptServer.instance.send_sync("executed", {
            "node": unique_id,
            "output": {
                "images": [{
                    "filename": preview_filename,
                    "subfolder": "",
                    "type": "temp"
                }]
            }
        })

    def generate(self, base_prompt, negative_prompt, prompt_list,
                 width, height, seed, steps, cfg, sampler, scheduler, smea, nai_model, save_prefix,
                 skip_indices="", variety=False, decrisper=False, free_only=True,
                 cfg_rescale=0.0, uncond_scale=1.0,
                 reference_image=None, char_ref_style_aware=True, char_ref_fidelity=1.0,
                 lut_name="None", lut_strength=0.3, enable_preview=True,
                 unique_id=None, prompt=None, extra_pnginfo=None):
        
        # free_only: Opus 무료 조건 (1MP 이하, 28 steps 이하)
        if free_only:
            pixel_limit = 1024 * 1024
            pixels = width * height
            if pixels > pixel_limit:
                # 비율 유지하면서 축소
                scale = (pixel_limit / pixels) ** 0.5
                width = int(width * scale) // 64 * 64
                height = int(height * scale) // 64 * 64
                print(f"[NAI MultiPrompt] free_only: Resolution adjusted to {width}x{height}")
            if steps > 28:
                print(f"[NAI MultiPrompt] free_only: Steps clamped from {steps} to 28")
                steps = 28
        
        # prompt_list 파싱
        blocks = prompt_list.strip().split("\n\n")
        lines = []
        for block in blocks:
            block_lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if not block_lines:
                continue
            
            first_content = None
            for line in block_lines:
                if not line.startswith("#"):
                    first_content = line
                    break
            
            if first_content and first_content.startswith("-"):
                continue
            
            merged = " ".join(
                line.strip() for line in block_lines 
                if not line.startswith("#")
            )
            if merged:
                lines.append(merged)
        
        # skip_indices 파싱
        skip_set = set()
        if skip_indices.strip():
            for part in skip_indices.split(","):
                part = part.strip()
                if part.isdigit():
                    skip_set.add(int(part))
        
        if not lines:
            raise ValueError("prompt_list가 비어있습니다.")
        
        # LUT 로드
        lut_data = None
        lut_size = 0
        if lut_name != "None":
            lut_path = find_lut_file(lut_name)
            if lut_path:
                lut_data, lut_size = parse_cube_lut(lut_path)
                print(f"[NAI MultiPrompt] LUT loaded: {lut_name}")
        
        all_images = []
        counter = self._get_next_counter(save_prefix)
        
        # 인터럽트 체크용
        import comfy.model_management as mm
        
        for idx, line in enumerate(lines):
            # 중단 체크
            mm.throw_exception_if_processing_interrupted()
            
            if (idx + 1) in skip_set:
                print(f"[NAI MultiPrompt] Skipping {idx + 1}/{len(lines)}: {line[:40]}...")
                continue
            
            print(f"[NAI MultiPrompt] Processing {idx + 1}/{len(lines)}: {line[:40]}...")
            
            first_tag = line.split(",")[0].strip().replace(" ", "_")
            filename = f"{idx + 1:02d}_{first_tag}"
            
            full_prompt = f"{base_prompt} {line}"
            
            # NAI API 호출
            current_seed = seed if seed != 0 else np.random.randint(0, 2**31 - 1)
            image = self.call_nai_api(
                full_prompt, negative_prompt, width, height,
                current_seed, steps, cfg, sampler, scheduler, smea,
                nai_model, variety, decrisper, cfg_rescale, uncond_scale,
                reference_image, char_ref_style_aware, char_ref_fidelity
            )
            
            # 프리뷰
            if enable_preview and unique_id is not None:
                self.send_preview(image, unique_id, idx, "nai")
            
            # LUT 적용
            if lut_data is not None and lut_strength > 0:
                image = apply_lut(image, lut_data, lut_size, lut_strength)
                if enable_preview and unique_id is not None:
                    self.send_preview(image, unique_id, idx, "final")
            
            # 저장
            self.save_image(image, save_prefix, filename, counter, prompt, extra_pnginfo)
            all_images.append(image)
        
        print(f"[NAI MultiPrompt] Complete! Generated {len(all_images)} images.")
        
        # 마지막 이미지를 UI에 표시
        if all_images and enable_preview:
            last_image = all_images[-1]
            import time
            temp_dir = folder_paths.get_temp_directory()
            timestamp = int(time.time() * 1000)
            preview_filename = f"nai_multiprompt_final_{timestamp}.png"
            preview_path = os.path.join(temp_dir, preview_filename)
            
            img = last_image[0].cpu().numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_img.save(preview_path, compress_level=1)
            
            return {"ui": {"images": [{"filename": preview_filename, "subfolder": "", "type": "temp"}]}, "result": (all_images,)}
        
        return {"ui": {"images": []}, "result": (all_images,)}

    def _get_next_counter(self, prefix):
        full_output_folder = os.path.join(self.output_dir, prefix)
        
        if not os.path.exists(full_output_folder):
            return 1
        
        existing_files = os.listdir(full_output_folder)
        if not existing_files:
            return 1
        
        max_counter = 0
        for f in existing_files:
            if f.endswith(".png"):
                try:
                    counter_str = f.rsplit("_", 1)[-1].replace(".png", "")
                    counter_val = int(counter_str)
                    max_counter = max(max_counter, counter_val)
                except:
                    pass
        
        return max_counter + 1


NODE_CLASS_MAPPINGS = {
    "MultiPromptGenerator": MultiPromptGenerator,
    "NAIMultiPromptGenerator": NAIMultiPromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPromptGenerator": "Multi Prompt Generator",
    "NAIMultiPromptGenerator": "NAI Multi Prompt Generator",
}
