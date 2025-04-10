import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
from torch import nn
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

####################### CONFIG #######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "stabilityai/stable-diffusion-2-1-base"
weights_path = "/data/lyee0001/GEN-AI/models_sd21_3/lora_unet_epoch5.pth"  
alpha = 1  
rank = 8  

test_faces_dir = "/data/lyee0001/GEN-AI/comic_faces_dataset/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/test_faces"
ground_truth_dir = "/data/lyee0001/GEN-AI/comic_faces_dataset/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/test_comics"
output_dir = "/data/lyee0001/GEN-AI/comic_faces_dataset/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/test_comics_generated_3"
os.makedirs(output_dir, exist_ok=True)

####################### LOAD BASE PIPELINE #######################
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
pipe = pipe.to(device)
unet = pipe.unet
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

####################### DEFINE LORA WRAPPER #######################
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank: int = 4, lora_alpha: float = 1.0):
        super().__init__()
        self.orig = orig_linear
        self.orig.weight.requires_grad_(False)
        if self.orig.bias is not None:
            self.orig.bias.requires_grad_(False)
        in_dim = orig_linear.in_features
        out_dim = orig_linear.out_features
        self.lora_down = nn.Linear(in_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_dim, bias=False)
        self.scale = lora_alpha
        nn.init.zeros_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.orig(x) + self.scale * self.lora_up(self.lora_down(x))

def apply_lora_to_unet(unet, rank=4, alpha=1.0):
    for name, module in unet.named_modules():
        # Look for modules that have all these attributes (typical for attention layers)
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v") and hasattr(module, "to_out"):
            module.to_q = LoRALinear(module.to_q, rank=rank, lora_alpha=alpha).to(device)
            module.to_k = LoRALinear(module.to_k, rank=rank, lora_alpha=alpha).to(device)
            module.to_v = LoRALinear(module.to_v, rank=rank, lora_alpha=alpha).to(device)
            if isinstance(module.to_out, nn.Sequential) and isinstance(module.to_out[0], nn.Linear):
                module.to_out[0] = LoRALinear(module.to_out[0], rank=rank, lora_alpha=alpha).to(device)
            elif isinstance(module.to_out, nn.Linear):
                module.to_out = LoRALinear(module.to_out, rank=rank, lora_alpha=alpha).to(device)

####################### APPLY LORA AND LOAD WEIGHTS #######################
apply_lora_to_unet(unet, rank=rank, alpha=alpha)
unet.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
print("LoRA weights loaded.")

####################### PREPARE TRANSFORM #######################
transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor()
])

####################### GET TEXT EMBEDDING #######################
with torch.no_grad():
    text_input = tokenizer(["comic style"], padding="max_length", truncation=True,
                           max_length=tokenizer.model_max_length, return_tensors="pt")
    prompt_embeds = text_encoder(text_input.input_ids.to(device))[0]

####################### EVALUATION LOOP #######################
test_face_paths = sorted([
    os.path.join(test_faces_dir, f)
    for f in os.listdir(test_faces_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

lpips_fn = lpips.LPIPS(net='vgg').to(device)
total_ssim, total_psnr, total_mse, total_lpips = 0.0, 0.0, 0.0, 0.0
num_evaluated = 0

################## Generation and evaluation #######################

for face_path in tqdm(test_face_paths, desc ="Processing images"):

    face_image = Image.open(face_path).convert("RGB")
    face_tensor = transform(face_image).unsqueeze(0).to(device)  
    face_tensor = face_tensor * 2.0 - 1.0  

    with torch.no_grad():
        latents = vae.encode(face_tensor).latent_dist.sample() * 0.18215
        latents = latents.clamp(-10, 10)
        
        timesteps = torch.full((latents.shape[0],), 1, dtype=torch.long, device=device)
        unet_out = unet(latents, timesteps, encoder_hidden_states=prompt_embeds).sample
        
        pred_latents = latents + alpha * unet_out
        
        pred_image = vae.decode(pred_latents / 0.18215).sample
        pred_image = (pred_image * 0.5 + 0.5).clamp(0, 1)
    
    base_name = os.path.basename(face_path)
    output_path = os.path.join(output_dir, base_name)
    save_image(pred_image, output_path)

    gt_path = os.path.join(ground_truth_dir, base_name)
    if os.path.exists(gt_path):
        gt_image = Image.open(gt_path).convert("RGB")
        gt_tensor = transform(gt_image).unsqueeze(0).to(device)
        gt_tensor = gt_tensor * 0.5 + 0.5
        
        pred_np = pred_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = gt_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        ssim_val = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1)
        psnr_val = peak_signal_noise_ratio(gt_np, pred_np, data_range=1)
        mse_val = torch.nn.functional.mse_loss(pred_image, gt_tensor).item()
        lpips_val = lpips_fn(pred_image, gt_tensor).mean().item()

        total_ssim += ssim_val
        total_psnr += psnr_val
        total_mse += mse_val
        total_lpips += lpips_val
        num_evaluated += 1

if num_evaluated > 0:
    print("====================================")
    print(f"Evaluated on {num_evaluated} image pairs")
    print(f"Average SSIM:  {total_ssim / num_evaluated:.4f}")
    print(f"Average PSNR:  {total_psnr / num_evaluated:.4f} dB")
    print(f"Average MSE:   {total_mse / num_evaluated:.6f}")
    print(f"Average LPIPS: {total_lpips / num_evaluated:.4f}")
    print("====================================")
else:
    print("No ground truth images were found for evaluation.")
