import os
import random
import argparse
from copy import deepcopy

import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
from huggingface_hub import hf_hub_url, cached_download

from kandinsky2 import CONFIG_2_1, Kandinsky2_1 


def parse_args():
    parser = argparse.ArgumentParser(description="Kandinsky 2.1 Textual Inversion training script.")
    parser.add_argument(
        "--data_root", type=str, default=None, required=True, help="A folder containing the training images."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="/tmp/kandinsky2",
        help="Path to pretrained model WITHOUT 2_1 folder",
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="textual-inversion",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="The resolution for input images, all the images will be resized to this size",
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader num workers.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--log_image_frequency", type=int, default=-1, help="How often save images. Values less zero - disable saving."
    )
    parser.add_argument(
        "--log_embed_frequency", type=int, default=500, help="How often save embeddings."
    )

    args = parser.parse_args()

    if args.data_root is None:
        raise ValueError("You must specify a train data directory.")

    return args

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

# Text templates and dataset class (with minor refactoring) from diffusers repo:
# https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        placeholder_token,
        img_size=512,
        learnable_property="object", # [object, style]
        flip_p=0.5,
        center_crop=False,
    ):
        self.data_root = data_root
        self.img_size = img_size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        example["text"] = text
        
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            h, w = img.shape[:2]
            min_side = min(h, w)
            img = img[(h - min_side) // 2 : (h + min_side) // 2, (w - min_side) // 2 : (w + min_side) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.img_size, self.img_size), resample=PIL.Image.Resampling.BICUBIC)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["image"] = torch.from_numpy(image).permute(2, 0, 1)
        return example
    
    
def download_models_if_not_exist(
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
):
    '''
    Download models in cache folder without model creation.
    '''
    cache_dir = os.path.join(cache_dir, "2_1")
    if task_type == "text2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name)
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=use_auth_token,
    )
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=prior_name)
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=prior_name,
        use_auth_token=use_auth_token,
    )
    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename=f"text_encoder/{name}")
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en,
            force_filename=name,
            use_auth_token=use_auth_token,
        )
    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename="movq_final.ckpt")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=use_auth_token,
    )
    config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.1", filename="ViT-L-14_stats.th")
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="ViT-L-14_stats.th",
        use_auth_token=use_auth_token,
    )


def add_noise(original_samples, noise, timesteps):
    num_diffusion_timesteps = 1000
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.00085
    beta_end = scale * 0.012
        
    betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=original_samples.dtype)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
        
    alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples


def generate_clip_emb(model,
        prompt,
        batch_size=1,
        prior_cf_scale=1,
        prior_steps="5",
        negative_prior_prompt="",
    ):
    prompts_batch = [prompt for _ in range(batch_size)]
    prior_cf_scales_batch = [prior_cf_scale] * len(prompts_batch)
    prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device=model.device)
    max_txt_length = model.prior.model.text_ctx
    tok, mask = model.tokenizer2.padded_tokens_and_mask(
        prompts_batch, max_txt_length
    )
    cf_token, cf_mask = model.tokenizer2.padded_tokens_and_mask(
        [negative_prior_prompt], max_txt_length
    )
    if not (cf_token.shape == tok.shape):
        cf_token = cf_token.expand(tok.shape[0], -1)
        cf_mask = cf_mask.expand(tok.shape[0], -1)
    tok = torch.cat([tok, cf_token], dim=0)
    mask = torch.cat([mask, cf_mask], dim=0)
    tok, mask = tok.to(device=model.device), mask.to(device=model.device)

    x = model.clip_model.token_embedding(tok).type(model.clip_model.dtype)
    x = x + model.clip_model.positional_embedding.type(model.clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND|
    x = model.clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.clip_model.ln_final(x).type(model.clip_model.dtype)
    txt_feat_seq = x
    txt_feat = (x[torch.arange(x.shape[0]), tok.argmax(dim=-1)] @ model.clip_model.text_projection)
    txt_feat, txt_feat_seq = txt_feat.float().to(model.device), txt_feat_seq.float().to(model.device)
    img_feat = model.prior(
        txt_feat,
        txt_feat_seq,
        mask,
        prior_cf_scales_batch,
        timestep_respacing=prior_steps,
    )
    return img_feat.to(model.model_dtype)


def save_embeds(model, save_path, placeholder_token, t1_place_token_id, t2_place_token_id):
    t1_embeds = model.text_encoder.model.transformer.get_input_embeddings().weight[t1_place_token_id]
    t2_embeds = model.clip_model.token_embedding.weight[t2_place_token_id]
    learned_embeds_dict = {
        't1': {
            placeholder_token: t1_embeds.cpu().detach(), 
        },
        't2':{
            placeholder_token: t2_embeds.cpu().detach(),
        },
    }
    torch.save(learned_embeds_dict, save_path)
    
    
def save_images(model, save_path, placeholder_token, img_size=512):
    simple_images = model.generate_text2img(
            f"a photo of a {placeholder_token}",
            num_steps=50, 
            batch_size=2, 
            guidance_scale=7.5,
            h=img_size, 
            w=img_size,
            sampler="p_sampler", 
            prior_cf_scale=4,
            prior_steps="5",
    )
    cool_images = model.generate_text2img(
            f"Professional high-quality art of a {placeholder_token}. photorealistic, 4k, HQ",
            num_steps=50, 
            batch_size=2, 
            guidance_scale=7.5,
            h=img_size, 
            w=img_size,
            sampler="p_sampler", 
            prior_cf_scale=4,
            prior_steps="5",
    )
    images = [*simple_images, *cool_images]
    gen_images = np.hstack([np.array(img) for img in images])
    Image.fromarray(gen_images).save(save_path)


def check_tokens_is_valid(model, placeholder_token, initializer_token):
    print("Check tokens...")
    if placeholder_token in model.tokenizer2.encoder: 
        raise ValueError(f"Word {placeholder_token} exists in tokenizer2. Please select another word.")

    if initializer_token not in model.tokenizer2.encoder:  
        raise ValueError(f"Word {initializer_token} doesn't exist in tokenizer2. Please select another word.")

    if len(model.tokenizer1.encode(placeholder_token)) == 3: 
        raise ValueError(f"Word {placeholder_token} exists in tokenizer1. Please select another word.")

    if len(model.tokenizer1.encode(initializer_token)) != 3: 
        raise ValueError(f"Word {initializer_token} doesn't exists in tokenizer1. Please select another word.")
    print("Selected tokens are correct")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    download_models_if_not_exist(task_type="text2img", cache_dir=args.cache_root)

    config = DictConfig(deepcopy(CONFIG_2_1))
    cache_dir = os.path.join(args.cache_root, "2_1")

    config["model_config"]["up"] = False
    config["model_config"]["use_fp16"] = False
    config["model_config"]["inpainting"] = False
    config["model_config"]["cache_text_emb"] = False
    config["model_config"]["use_flash_attention"] = False

    config["tokenizer_name"] = os.path.join(cache_dir, "text_encoder")
    config["text_enc_params"]["model_path"] = os.path.join(cache_dir, "text_encoder")
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")

    model_path = os.path.join(cache_dir, "decoder_fp16.ckpt")
    prior_path = os.path.join(cache_dir, "prior_fp16.ckpt")

    # Create model and check that initializer_token exists in both tokenizers and 
    # placeholder_token doesn't exitst in both tokenizers
    model = Kandinsky2_1(config, model_path, prior_path, args.device, task_type="text2img")
    check_tokens_is_valid(model, args.placeholder_token, args.initializer_token)

    # Convert the initializer_token and placeholder_token to tokenizer1 ids 
    model.tokenizer1.add_tokens(args.placeholder_token)

    t1_init_token_id = model.tokenizer1.encode(args.initializer_token, add_special_tokens=False)[0]
    t1_place_token_id = model.tokenizer1.convert_tokens_to_ids(args.placeholder_token)

    # Extend embedding size
    model.text_encoder.model.transformer.resize_token_embeddings(len(model.tokenizer1))
    
    # Initialise new placeholder weights with the embeddings of the initializer token
    token_embeds = model.text_encoder.model.transformer.get_input_embeddings().weight.data
    token_embeds[t1_place_token_id] = token_embeds[t1_init_token_id]
    
    # Convert the initializer_token, placeholder_token to ids for tokenizer2
    # and add placeholder_token to tokenizer2
    t2p_index_to_add = len(model.tokenizer2.encoder)
    model.tokenizer2.encoder[args.placeholder_token] = t2p_index_to_add
    model.tokenizer2.decoder[t2p_index_to_add] = args.placeholder_token
    model.tokenizer2.cache[args.placeholder_token] = args.placeholder_token

    t2_place_token_id = model.tokenizer2.encode(args.placeholder_token)[0]
    t2_init_token_id = model.tokenizer2.encode(args.initializer_token)[0]

    old_vocab_size, t2_embed_size = model.clip_model.token_embedding.weight.shape

    # Create new embeddings 
    # Copy old weights to the new embeddings and initialize new token 
    new_embed = nn.Embedding(old_vocab_size + 1, t2_embed_size).to(args.device)
    new_embed.weight.data[:old_vocab_size, :] = model.clip_model.token_embedding.weight.data.clone()
    new_embed.weight.data[t2_place_token_id, :] = new_embed.weight.data[t2_init_token_id, :]

    model.clip_model.token_embedding = deepcopy(new_embed)

    ## Freeze all except embeddings
    model.image_encoder.requires_grad_(False)
    model.model.requires_grad_(False)
    model.prior.requires_grad_(False)

    model.clip_model.token_embedding.requires_grad_(True)
    model.clip_model.transformer.requires_grad_(False)

    model.text_encoder.model.transformer.get_input_embeddings().requires_grad_(True)
    model.text_encoder.model.transformer.embeddings.position_embeddings.requires_grad_(False)
    model.text_encoder.model.transformer.embeddings.token_type_embeddings.requires_grad_(False)
    model.text_encoder.model.transformer.encoder.requires_grad_(False)
    model.text_encoder.model.transformer.pooler.requires_grad_(False)
    model.text_encoder.model.LinearTransformation.requires_grad_(False)

    # Configure optimizer, dataset and dataloader
    optimizer = torch.optim.AdamW(
        list(model.text_encoder.model.transformer.get_input_embeddings().parameters()) +
        list(model.clip_model.token_embedding.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = TextualInversionDataset(
        data_root=args.data_root,
        placeholder_token=args.placeholder_token,
        img_size=args.img_size,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    # Save original embeddings from both models
    orig_t1_params = model.text_encoder.model.transformer.get_input_embeddings().weight.data.clone()
    orig_t2_params = model.clip_model.token_embedding.weight.data.clone()
    
    weight_dtype = model.model.dtype
    progress_bar_epochs = tqdm(range(1, args.epochs + 1))

    for epoch in progress_bar_epochs:
        model.text_encoder.train()
        model.clip_model.train()
        for batch in train_dataloader:
            model_kwargs = {}
            # Convert images to latent representation and add noise
            latents = model.image_encoder.encode(batch["image"].to(device=args.device, dtype=weight_dtype)).detach()
            latents = latents * model.scale

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (args.train_batch_size,), device=latents.device).long()

            noisy_latents = add_noise(latents, noise, timesteps).to(weight_dtype)

            # Get hidden parameters for both models
            # First Model Parameters
            image_emb = generate_clip_emb(
                model,
                batch["text"][0],
                batch_size=args.train_batch_size,
                prior_cf_scale=4,
                prior_steps="5",
                negative_prior_prompt="",
            )
            
            model_kwargs["image_emb"] = image_emb.to(weight_dtype)

            # Second Model Parameters
            tokens = model.tokenizer1(
                batch["text"][0],
                padding="max_length",
                truncation=True,
                max_length=77,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            model_kwargs["full_emb"], model_kwargs["pooled_emb"] = model.text_encoder(
                tokens=tokens['input_ids'].long().to(device=args.device), 
                mask=tokens['attention_mask'].to(device=args.device),
            )

            model_kwargs["full_emb"] = model_kwargs["full_emb"].to(weight_dtype) 
            model_kwargs["pooled_emb"] = model_kwargs["pooled_emb"].to(weight_dtype) 

            # Predict noise obviously
            model_pred = model.model(noisy_latents, timesteps, **model_kwargs)[:, :4]

            loss = F.mse_loss(model_pred.float(), noise, reduction="mean")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # We don't need update all embeddings weights. Only new embeddings.
            with torch.no_grad():
                index_no_updates_t1 = torch.arange(len(model.tokenizer1)) != t1_place_token_id
                model.text_encoder.model.transformer.get_input_embeddings().weight[
                    index_no_updates_t1
                ] = orig_t1_params[index_no_updates_t1]
                
                index_no_updates_t2 = torch.arange(model.clip_model.token_embedding.weight.shape[0]) != t2_place_token_id
                model.clip_model.token_embedding.weight[
                    index_no_updates_t2
                ] = orig_t2_params[index_no_updates_t2]
                
            progress_bar_epochs.set_postfix(**{"loss": loss.cpu().detach().item()})
        
        if epoch % args.log_embed_frequency == 0:
            embed_save_path = os.path.join(args.output_dir, f"{epoch}_epoch_embeds.bin")
            save_embeds(model, embed_save_path, args.placeholder_token, t1_place_token_id, t2_place_token_id)

        if args.log_image_frequency > 0 and (epoch % args.log_image_frequency == 0):
            images_root = os.path.join(args.output_dir, "images")
            os.makedirs(images_root, exist_ok=True)
            image_save_path = os.path.join(images_root, f"{epoch}_epoch_images.jpg")
            save_images(model, image_save_path, args.placeholder_token)    
    

if __name__ == "__main__":
    main()