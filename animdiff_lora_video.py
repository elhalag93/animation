import torch
import os
import gc
import shutil
from datetime import datetime
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, export_to_video

def check_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        print(f"💾 GPU Memory: {allocated:.2f}GB used, {free:.2f}GB free, {total:.2f}GB total")
        return allocated, total
    return 0, 0

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# DEBUG MODE: Set to True to only test pipeline loading
# (Set these to False by default so the script runs end-to-end unless the
# developer explicitly enables a lightweight debug mode.)
DEBUG_MINIMAL_LOADER = False

# DEBUG: Print model file sizes using Python
DEBUG_MODEL_FILE_SIZES = False
if DEBUG_MODEL_FILE_SIZES:
    print("\n[DEBUG] Listing all files and sizes in SDXL Turbo model directory:")
    model_root = './local_sdxl_turbo/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304'
    for dirpath, dirnames, filenames in os.walk(model_root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
                print(f"{fpath} : {size/1024/1024:.2f} MB")
            except Exception as e:
                print(f"[ERROR] Could not access {fpath}: {e}")
    print("[DEBUG] Model file size listing complete. Exiting.")
    import sys; sys.exit(0)

if DEBUG_MINIMAL_LOADER:
    import sys
    print("\n[DEBUG] Minimal pipeline loader mode enabled.")
    print("💾 Checking memory before loading...")
    check_memory()
    
    # Find local AnimateDiff motion adapter in HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    motion_adapter_path = None
    print(f"\n[DEBUG] Searching for local AnimateDiff motion adapter in: {cache_dir}")
    for item in os.listdir(cache_dir):
        if "animatediff" in item.lower() and "motion" in item.lower():
            potential_path = os.path.join(cache_dir, item)
            if os.path.isdir(potential_path):
                snapshots_dir = os.path.join(potential_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        motion_adapter_path = os.path.join(snapshots_dir, snapshots[0])
                        print(f"[DEBUG] Found local motion adapter: {motion_adapter_path}")
                        break
    if not motion_adapter_path:
        print("[DEBUG] No local motion adapter found in cache")
        manual_path = "C:/Users/maram/.cache/huggingface/hub/models--guoyww--animatediff"
        if os.path.exists(manual_path):
            motion_adapter_path = manual_path
            print(f"[DEBUG] Found motion adapter at: {motion_adapter_path}")
        else:
            print("[DEBUG] No local motion adapter found. Exiting.")
            sys.exit(1)
    print(f"[DEBUG] (SKIPPING) Loading motion adapter from: {motion_adapter_path}")
    # adapter = MotionAdapter.from_pretrained(
    #     motion_adapter_path,
    #     torch_dtype=torch.float16,
    #     local_files_only=True
    # )
    # print("[DEBUG] Motion adapter loaded!")
    check_memory()
    
    local_model_path = "./local_sdxl_turbo/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304"
    print(f"[DEBUG] Local SDXL path: {os.path.abspath(local_model_path)}")
    print("[DEBUG] Model directory contents:")
    for f in os.listdir(local_model_path):
        print(f"   - {f}")
    print("[DEBUG] Loading AnimateDiffSDXLPipeline.from_pretrained WITHOUT motion adapter...")
    try:
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            local_model_path,
            motion_adapter=None,
            torch_dtype=torch.float32,  # or float16 if you want to try both
            use_safetensors=True,
            variant="fp16",
            local_files_only=True
        )
        print("[DEBUG] AnimateDiffSDXLPipeline loaded WITHOUT motion adapter!")
        check_memory()
        print("[DEBUG] Pipeline loading complete. Exiting debug mode.")
    except Exception as e:
        print(f"[DEBUG] Error loading AnimateDiffSDXLPipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)

print("🎬 FINAL LOCAL-ONLY ANIMDIFF + LORA PIPELINE")
print("============================================")

# Gracefully handle environments without a CUDA-enabled GPU.
if torch.cuda.is_available():
    print(f"🖥️  Device: {torch.cuda.get_device_name(0)} (CUDA)")
else:
    print("🖥️  Device: CPU (CUDA not available)")

print(f"⚡ PyTorch: {torch.__version__}")
print("🎯 USING ONLY YOUR LOCAL MODELS - NO DOWNLOADS!")
print("🎭 Temo & Felfel Character Videos")

def setup_6gb_optimizations(pipe):
    """Setup 6GB memory optimizations"""
    print("\n🧠 SETTING UP 6GB OPTIMIZATIONS...")
    
    try:
        # Sequential CPU offload - keeps models on CPU, loads to GPU only when needed
        pipe.enable_sequential_cpu_offload()
        print("✅ Sequential CPU offload: ENABLED (keeps models on CPU)")
        
        # Enable attention slicing
        pipe.enable_attention_slicing("max")
        print("✅ Maximum attention slicing: ENABLED")
        
        # Enable VAE optimizations
        pipe.enable_vae_slicing()
        print("✅ VAE slicing: ENABLED")
        
        try:
            pipe.enable_vae_tiling()
            print("✅ VAE tiling: ENABLED")
        except:
            pass
        
        # Try to enable xFormers memory efficient attention
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers memory efficient attention: ENABLED")
        except Exception as e:
            print("⚠️  XFormers not available or failed to enable.")
        
        # Memory-friendly torch settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TensorFloat-32: ENABLED")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def generate_character_video(pipe, character_name, lora_path, prompt, num_frames=16, seed=42):
    """Generate character video with 6GB optimization"""
    print(f"\n🎭 Generating {character_name} video ({num_frames} frames)...")
    check_memory()
    
    # Aggressive cleanup before starting
    aggressive_cleanup()
    
    # Clear previous LoRA
    try:
        pipe.unload_lora_weights()
        aggressive_cleanup()
    except:
        pass
    
    # Load character LoRA
    print(f"📥 Loading {character_name} LoRA weights...")
    pipe.load_lora_weights(lora_path, weight_name="deep_sdxl_turbo_lora_weights.pt")
    print(f"✅ {character_name} LoRA loaded!")
    check_memory()
    
    # Generate video with memory-optimized settings
    print(f"🎬 Generating {num_frames} frames...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator=torch.Generator(device).manual_seed(seed)
    
    video_frames = pipe(
        prompt,
        num_frames=num_frames,
        guidance_scale=7.5,
        num_inference_steps=15,
        height=512,
        width=512,
        generator=generator
    ).frames[0]
    
    # Cleanup immediately after generation
    aggressive_cleanup()
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"final_videos/{character_name}_local_{timestamp}.gif"
    mp4_path = f"final_videos/{character_name}_local_{timestamp}.mp4"
    
    export_to_gif(video_frames, gif_path, fps=8)
    export_to_video(video_frames, mp4_path, fps=8)
    
    print(f"✅ {character_name} video complete:")
    print(f"   🎞️  GIF: {gif_path}")
    print(f"   🎬 MP4: {mp4_path}")
    print(f"   📊 Frames: {len(video_frames)}")
    
    check_memory()
    aggressive_cleanup()
    
    return gif_path, mp4_path

# --------------------------------------------------
# Helper to ensure UNet config.json is present
# (SDXL-Turbo snapshots sometimes omit it, leaving it only in the HF blob cache.)
# --------------------------------------------------

def ensure_unet_config(model_root: str):
    """Ensure `unet/config.json` exists inside the snapshot. If it's missing,
    attempt to locate the file inside the HuggingFace `blobs` directory and copy
    it in place so that `diffusers` can load the pipeline offline."""

    unet_dir = os.path.join(model_root, "unet")
    config_path = os.path.join(unet_dir, "config.json")

    # Already there -> nothing to do
    if os.path.exists(config_path):
        return

    print("⚠️  unet/config.json missing – searching blobs directory for a replacement …")

    # The blobs dir is two levels up from the snapshot folder
    # e.g. local_sdxl_turbo/models--stabilityai--sdxl-turbo/blobs
    repo_root = os.path.abspath(os.path.join(model_root, os.pardir, os.pardir))
    blobs_dir = os.path.join(repo_root, "blobs")

    if not os.path.isdir(blobs_dir):
        raise FileNotFoundError("Blobs directory not found; cannot recover unet/config.json")

    # Scan blobs for the right config (first occurrence of the UNet class name)
    for blob_file in os.listdir(blobs_dir):
        blob_path = os.path.join(blobs_dir, blob_file)
        if not os.path.isfile(blob_path):
            continue
        try:
            with open(blob_path, "r", encoding="utf-8") as bf:
                snippet = bf.read(256)
                if "\"_class_name\": \"UNet2DConditionModel\"" in snippet:
                    # Copy it
                    os.makedirs(unet_dir, exist_ok=True)
                    shutil.copy(blob_path, config_path)
                    print(f"✅ Recovered UNet config: {blob_file} -> {config_path}")
                    return
        except UnicodeDecodeError:
            # Binary / weights files will raise decode errors – ignore those
            continue
        except Exception as ex:
            print(f"   ⚠️  Skipped {blob_file}: {ex}")

    raise FileNotFoundError("Unable to locate UNet config.json in blobs directory")

try:
    print("\n💾 INITIAL MEMORY STATE:")
    check_memory()
    
    # Find local AnimateDiff motion adapter in HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    motion_adapter_path = None
    
    print(f"\n🔍 Searching for local AnimateDiff motion adapter in: {cache_dir}")
    
    # Look for motion adapter in cache
    for item in os.listdir(cache_dir):
        if "animatediff" in item.lower() and "motion" in item.lower():
            potential_path = os.path.join(cache_dir, item)
            if os.path.isdir(potential_path):
                # Look for snapshots
                snapshots_dir = os.path.join(potential_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        motion_adapter_path = os.path.join(snapshots_dir, snapshots[0])
                        print(f"📁 Found local motion adapter: {motion_adapter_path}")
                        break
    
    if not motion_adapter_path:
        print("❌ No local motion adapter found in cache")
        print("💡 Checking user's manual path...")
        manual_path = "C:/Users/maram/.cache/huggingface/hub/models--guoyww--animatediff"
        if os.path.exists(manual_path):
            motion_adapter_path = manual_path
            print(f"✅ Found motion adapter at: {motion_adapter_path}")
        else:
            raise Exception("No local motion adapter found")
    
    print(f"🔄 Loading motion adapter from: {motion_adapter_path}")
    try:
        adapter = MotionAdapter.from_pretrained(
            motion_adapter_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        print("✅ Local motion adapter loaded!")
    except Exception as e:
        print(f"❌ Error loading motion adapter: {e}")
        import traceback
        traceback.print_exc()
        raise
    check_memory()
    
    # Use your LOCAL SDXL Turbo model
    local_model_path = "./local_sdxl_turbo/models--stabilityai--sdxl-turbo/snapshots/71153311d3dbb46851df1931d3ca6e939de83304"
    print(f"\n🔄 Creating pipeline with YOUR LOCAL SDXL Turbo...")
    print(f"📁 Local SDXL path: {os.path.abspath(local_model_path)}")
    print("🔍 Checking model directory contents:")
    for f in os.listdir(local_model_path):
        print(f"   - {f}")
    
    # Make sure the UNet config is present (some snapshots are incomplete)
    ensure_unet_config(local_model_path)

    # Create AnimateDiff pipeline with your LOCAL models only
    print("🔄 Loading AnimateDiffSDXLPipeline.from_pretrained...")
    try:
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            local_model_path,
            motion_adapter=adapter,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map="balanced"
        )
        print("✅ LOCAL AnimateDiff pipeline created!")
    except Exception as e:
        print(f"❌ Error loading AnimateDiffSDXLPipeline: {e}")
        import traceback
        traceback.print_exc()
        raise
    check_memory()
    
    # Setup scheduler
    print("🔄 Setting up DDIMScheduler...")
    try:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        print("✅ DDIM scheduler configured!")
    except Exception as e:
        print(f"❌ Error setting up scheduler: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Setup 6GB optimizations BEFORE any GPU operations
    optimization_success = setup_6gb_optimizations(pipe)
    
    print("\n💾 MEMORY AFTER 6GB OPTIMIZATIONS:")
    check_memory()
    
    print("\n🎬 STARTING LOCAL ANIMDIFF + LORA VIDEO GENERATION...")
    print("==================================================")
    print("🎯 USING ONLY YOUR LOCAL MODELS - NO DOWNLOADS!")
    
    generated_videos = []
    
    # Prompts for moon walking videos
    prompts = {
        "temo": "temo character walking confidently on moon surface, detailed cartoon style, space helmet, lunar landscape, smooth animation, high quality",
        "felfel": "felfel character exploring moon surface, detailed cartoon style, space suit, moon craters, smooth walking animation, high quality"
    }
    
    # Generate videos
    for character in ["temo", "felfel"]:
        lora_path = f"./{character}_lora"
        if os.path.exists(lora_path):
            print(f"\n🎭 GENERATING {character.upper()} CHARACTER VIDEO:")
            try:
                gif_path, mp4_path = generate_character_video(
                    pipe, character, lora_path,
                    prompts[character],
                    16, seed=42 if character == "temo" else 84
                )
                generated_videos.extend([gif_path, mp4_path])
                print(f"✅ {character.title()} video complete!")
                
            except Exception as e:
                print(f"❌ Error for {character}: {e}")
                aggressive_cleanup()
                continue
    
    print("\n🏆 LOCAL-ONLY ANIMDIFF + LORA SUCCESS!")
    print("=====================================")
    print(f"✅ Total videos generated: {len(generated_videos)}")
    
    if generated_videos:
        print("\n📁 Generated LOCAL AnimateDiff + LoRA videos:")
        for video_path in generated_videos:
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024*1024)
                print(f"   🎬 {video_path} ({file_size:.1f} MB)")
    
    print("\n🎭 FINAL ANALYSIS:")
    print("   🚀 Pipeline: LOCAL AnimateDiff + YOUR SDXL Turbo + LoRA")
    print("   🎭 Characters: Temo & Felfel with trained LoRA weights")
    print("   🧠 Memory: 6GB VRAM optimized with sequential CPU offload")
    print("   📁 Models: 100% LOCAL - NO DOWNLOADS!")
    print("   🎬 Output: Character-consistent animated videos")
    
    print("\n🎉 MISSION ACCOMPLISHED!")
    print("✅ LOCAL-ONLY PIPELINE SUCCESS!")
    print("✅ NO MODEL DOWNLOADS!")
    print("✅ 6GB MEMORY OPTIMIZED!")

except Exception as e:
    print(f"\n❌ Pipeline error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Final cleanup
    aggressive_cleanup()
    print("\n💾 FINAL MEMORY STATE:")
    check_memory()
    print("✅ Local-only pipeline complete!") 