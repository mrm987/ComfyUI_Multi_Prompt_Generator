import os
import glob
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
    
    return np.array(lut_data, dtype=np.float32), lut_size


def apply_lut(image, lut_data, lut_size, strength=1.0):
    """Apply 3D LUT to image"""
    if lut_data is None:
        return image
    
    # image: [B, H, W, C] tensor, 0-1 range
    img_np = image.cpu().numpy()
    result = np.zeros_like(img_np)
    
    for b in range(img_np.shape[0]):
        img = img_np[b]
        h, w, c = img.shape
        
        # Reshape LUT
        lut_3d = lut_data.reshape((lut_size, lut_size, lut_size, 3))
        
        # Scale image to LUT indices
        img_scaled = img * (lut_size - 1)
        
        # Get integer indices and fractions for trilinear interpolation
        idx_low = np.floor(img_scaled).astype(np.int32)
        idx_high = np.ceil(img_scaled).astype(np.int32)
        frac = img_scaled - idx_low
        
        # Clamp indices
        idx_low = np.clip(idx_low, 0, lut_size - 1)
        idx_high = np.clip(idx_high, 0, lut_size - 1)
        
        # Trilinear interpolation
        r_low, g_low, b_low = idx_low[:,:,0], idx_low[:,:,1], idx_low[:,:,2]
        r_high, g_high, b_high = idx_high[:,:,0], idx_high[:,:,1], idx_high[:,:,2]
        r_frac, g_frac, b_frac = frac[:,:,0:1], frac[:,:,1:2], frac[:,:,2:3]
        
        # 8 corners of the cube
        c000 = lut_3d[r_low, g_low, b_low]
        c001 = lut_3d[r_low, g_low, b_high]
        c010 = lut_3d[r_low, g_high, b_low]
        c011 = lut_3d[r_low, g_high, b_high]
        c100 = lut_3d[r_high, g_low, b_low]
        c101 = lut_3d[r_high, g_low, b_high]
        c110 = lut_3d[r_high, g_high, b_low]
        c111 = lut_3d[r_high, g_high, b_high]
        
        # Interpolate
        c00 = c000 * (1 - r_frac) + c100 * r_frac
        c01 = c001 * (1 - r_frac) + c101 * r_frac
        c10 = c010 * (1 - r_frac) + c110 * r_frac
        c11 = c011 * (1 - r_frac) + c111 * r_frac
        
        c0 = c00 * (1 - g_frac) + c10 * g_frac
        c1 = c01 * (1 - g_frac) + c11 * g_frac
        
        lut_result = c0 * (1 - b_frac) + c1 * b_frac
        
        # Blend with original based on strength
        result[b] = img * (1 - strength) + lut_result * strength
    
    result = np.clip(result, 0, 1)
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
                "base_prompt": ("STRING", {"multiline": True, "default": "1girl, solo,"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "lowres, bad quality,"}),
                "prompt_list": ("STRING", {"multiline": True, "default": "smile, happy\nangry, furrowed brow\nsad, crying"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "enable_upscale": ("BOOLEAN", {"default": True}),
                "save_prefix": ("STRING", {"default": "MultiPrompt"}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "scale_factor": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "step": 0.01}),
                "upscale_steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "upscale_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "upscale_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "size_alignment": (["64", "8", "none"],),
                "lut_name": (get_lut_files(),),
                "lut_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
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

    def do_sample(self, model, seed, steps, cfg, positive, negative, latent_image, denoise=1.0):
        """샘플링 실행"""
        device = comfy.model_management.get_torch_device()
        latent = latent_image.clone().to(device)
        
        noise = comfy.sample.prepare_noise(latent, seed, None)
        
        samples = comfy.sample.sample(
            model, 
            noise, 
            steps, 
            cfg, 
            "euler_ancestral",
            "normal",
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

    def save_image(self, image, prefix, filename, counter):
        """이미지 저장"""
        full_output_folder = os.path.join(self.output_dir, prefix)
        os.makedirs(full_output_folder, exist_ok=True)
        
        file = f"{filename}_{counter:05d}.png"
        filepath = os.path.join(full_output_folder, file)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(filepath, compress_level=4)
        
        print(f"[MultiPrompt] Saved: {filepath}")
        return filepath

    def send_preview(self, image, unique_id, idx):
        """생성 중 프리뷰를 UI로 전송"""
        temp_dir = folder_paths.get_temp_directory()
        
        # temp 폴더에 프리뷰 저장
        preview_filename = f"multiprompt_preview_{unique_id}_{idx}.png"
        preview_path = os.path.join(temp_dir, preview_filename)
        
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img.save(preview_path, compress_level=4)
        
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

    def generate(self, model, clip, vae, latent, base_prompt, negative_prompt, prompt_list,
                 seed, steps, cfg, enable_upscale, save_prefix,
                 upscale_model=None, scale_factor=0.7, upscale_steps=15, 
                 upscale_cfg=5.0, upscale_denoise=0.5, size_alignment="64",
                 lut_name="None", lut_strength=0.3, enable_preview=True, unique_id=None):
        
        # prompt_list 파싱
        lines = [line.strip() for line in prompt_list.strip().split("\n") if line.strip()]
        
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
        
        for idx, line in enumerate(lines):
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
                model, seed, steps, cfg,
                positive_cond, negative_cond,
                latent_samples, denoise=1.0
            )
            image = self.decode_vae(vae, samples)
            
            # 1차 결과 프리뷰 전송
            if enable_preview and unique_id is not None:
                self.send_preview(image, unique_id, idx)
            
            # 2단계: 업스케일
            if enable_upscale and upscale_model is not None:
                # 업스케일 모델 적용
                upscaled = self.upscale_with_model(upscale_model, image)
                
                # 64배수로 리사이즈
                resized = self.resize_with_alignment(upscaled, scale_factor, size_alignment)
                
                # VAE 인코딩 → 2차 샘플링 → 디코딩
                latent_up = self.encode_vae(vae, resized)
                
                samples_up = self.do_sample(
                    model, seed, upscale_steps, upscale_cfg,
                    positive_cond, negative_cond,
                    latent_up, denoise=upscale_denoise
                )
                image = self.decode_vae(vae, samples_up)
            
            # 3단계: LUT 적용
            if lut_data is not None and lut_strength > 0:
                image = apply_lut(image, lut_data, lut_size, lut_strength)
            
            # 최종 결과 프리뷰 교체
            if enable_preview and unique_id is not None:
                self.send_preview(image, unique_id, idx)
            
            # 저장
            self.save_image(image, save_prefix, filename, counter)
            all_images.append(image)
        
        print(f"[MultiPrompt] Complete! Generated {len(all_images)} images.")
        
        # 마지막 프리뷰 정보 반환 (UI에 유지)
        ui_images = []
        if enable_preview and unique_id is not None and len(lines) > 0:
            last_preview = f"multiprompt_preview_{unique_id}_{len(lines)-1}.png"
            ui_images = [{"filename": last_preview, "subfolder": "", "type": "temp"}]
        
        return {"ui": {"images": ui_images}, "result": (all_images,)}

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


NODE_CLASS_MAPPINGS = {
    "MultiPromptGenerator": MultiPromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiPromptGenerator": "Multi Prompt Generator"
}
