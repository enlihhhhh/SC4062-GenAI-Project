import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast, GradScaler
import lpips

####################### CONFIG #######################
model_name = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 768  # SD 2.1 default
batch_size = 4
num_epochs = 5
learning_rate = 1e-5
prompt_text = "comic style"
rank = 8
alpha = 1.0
log_interval = 50
model_dir = "models_sd21_3"
plot_dir = "plots_sd21_3"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

######################## DATALOADER #######################
class ComicPromptDataset(Dataset):
    def __init__(self, real_dir, comic_dir, prompt, transform):
        self.real_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith((".jpg", ".png"))])
        self.comic_paths = sorted([os.path.join(comic_dir, f) for f in os.listdir(comic_dir) if f.endswith((".jpg", ".png"))])
        assert len(self.real_paths) == len(self.comic_paths)
        self.prompt = prompt
        self.transform = transform

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self, idx):
        real = Image.open(self.real_paths[idx]).convert("RGB")
        comic = Image.open(self.comic_paths[idx]).convert("RGB")
        return self.transform(real), self.transform(comic), self.prompt

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
pipe.to(device)

vae = pipe.vae
unet = pipe.unet
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder

vae.requires_grad_(False)
text_encoder.requires_grad_(False)

####################### LORA #######################
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, lora_alpha):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.scale = lora_alpha
        nn.init.zeros_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.lora_down(x))

def apply_lora_to_unet(unet, rank, alpha):
    for name, module in unet.named_modules():
        if hasattr(module, "to_q") and hasattr(module, "to_k") and hasattr(module, "to_v") and hasattr(module, "to_out"):
            module.to_q = LoRALinear(module.to_q, rank, alpha).to(device)
            module.to_k = LoRALinear(module.to_k, rank, alpha).to(device)
            module.to_v = LoRALinear(module.to_v, rank, alpha).to(device)
            if isinstance(module.to_out, nn.Sequential) and isinstance(module.to_out[0], nn.Linear):
                module.to_out[0] = LoRALinear(module.to_out[0], rank, alpha).to(device)
            elif isinstance(module.to_out, nn.Linear):
                module.to_out = LoRALinear(module.to_out, rank, alpha).to(device)

apply_lora_to_unet(unet, rank=rank, alpha=alpha)
trainable_params = [p for p in unet.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable_params)}")

lpips_fn = lpips.LPIPS(net='vgg').to(device)

####################### DATA #######################
real_dir = "comic_faces_dataset/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/faces"
comic_dir = "comic_faces_dataset/face2comics_v2.0.0_by_Sxela/face2comics_v2.0.0_by_Sxela/comics"
dataset = ComicPromptDataset(real_dir, comic_dir, prompt_text, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"Loaded {len(dataset)} image pairs.")

####################### TRAINING #######################
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
scaler = GradScaler()

step_losses, epoch_losses = [], []

for epoch in range(1, num_epochs + 1):
    unet.train()
    epoch_loss = 0.0

    for step, (real_img, comic_img, prompts) in enumerate(tqdm(loader, desc=f"Epoch {epoch}"), start=1):
        real_img = real_img.to(device) * 2 - 1
        comic_img = comic_img.to(device) * 2 - 1

        with torch.no_grad():
            text_input = tokenizer(list(prompts), padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
            prompt_embeds = text_encoder(text_input)[0]
            real_latents = vae.encode(real_img).latent_dist.sample() * 0.18215
            comic_latents = vae.encode(comic_img).latent_dist.sample() * 0.18215

        optimizer.zero_grad()
        with autocast():
            t = torch.randint(1, 10, (real_latents.shape[0],), device=device)
            noise_pred = unet(real_latents, t, encoder_hidden_states=prompt_embeds).sample
            predicted_latents = real_latents + alpha * noise_pred
            decoded_imgs = vae.decode(predicted_latents / 0.18215).sample
            decoded_imgs = (decoded_imgs * 0.5 + 0.5).clamp(0, 1)
            comic_img_clamped = (comic_img * 0.5 + 0.5).clamp(0, 1)

            mse = nn.functional.mse_loss(decoded_imgs, comic_img_clamped)
            perceptual = lpips_fn(decoded_imgs, comic_img_clamped).mean()
            loss = mse + 0.8 * perceptual

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_losses.append(loss.item())
        epoch_loss += loss.item()

        if step % log_interval == 0:
            print(f"Epoch {epoch} Step {step}: Loss {loss.item():.4f} (MSE {mse.item():.4f} + LPIPS {perceptual.item():.4f})")

    avg_epoch_loss = epoch_loss / len(loader)
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_epoch_loss:.4f}")

    torch.save(unet.state_dict(), os.path.join(model_dir, f"lora_unet_epoch{epoch}.pth"))

####################### PLOT #######################
plt.figure(figsize=(6, 4))
plt.plot(step_losses, label="Loss per step", alpha=0.5)
plt.plot([i * len(loader) for i in range(1, num_epochs + 1)], epoch_losses, label="Loss per epoch", marker="o")
plt.title("Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Total Loss (MSE + LPIPS)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "loss_curve_sd21.jpeg"), dpi=300)