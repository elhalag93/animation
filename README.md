# AnimDiff Character Mixer 🎬

A high-quality animation generation system that integrates custom LoRA character models (Temo and Felfel) with AnimDiff to create distinct mixed cartoon animations using xFormers optimization.

## Features ✨

- **High-Quality Output**: Maximum quality settings with 512x512 resolution, 50 inference steps
- **Character Mixing**: Blend Temo and Felfel characters with customizable strength ratios
- **xFormers Optimization**: Memory-efficient attention for faster generation
- **Multiple Presets**: Pre-configured mixing ratios for different character combinations
- **Batch Processing**: Generate multiple animations with different settings
- **Professional Pipeline**: Uses EulerDiscreteScheduler and advanced optimizations

## Prerequisites 📋

### Required Packages (Already Installed)
- `torch` (2.0.1+cu118)
- `diffusers` (0.18.2)
- `transformers` (4.30.2)
- `xformers` (0.0.20)
- `accelerate` (0.20.3)
- `safetensors` (0.5.3)
- `PyYAML`
- `Pillow`
- `numpy`

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ system RAM
- 10GB+ free disk space

## Project Structure 📁

```
last/
├── temo_sdxl_turbo_lora/
│   ├── sdxl_turbo_lora_weights.pt     # Temo character LoRA
│   └── training_info.json
├── felfel_sdxl_turbo_lora/
│   ├── sdxl_turbo_lora_weights.pt     # Felfel character LoRA
│   └── training_info.json
├── animdiff_character_mixer.py        # Main mixer script
├── enhanced_animdiff_mixer.py         # Enhanced version with config
├── test_animdiff.py                   # Test script
├── config.yaml                        # Configuration file
└── README.md                          # This file
```

## Quick Start 🚀

### 1. Test the Setup

First, run the test script to verify everything is working:

```bash
python test_animdiff.py
```

This will:
- Test AnimDiff pipeline initialization
- Inspect your LoRA models
- Test character blending
- Generate a test animation

### 2. Basic Animation Generation

```python
from animdiff_character_mixer import AnimDiffCharacterMixer

# Initialize the mixer
mixer = AnimDiffCharacterMixer()

# Generate a balanced mix animation
output_path = mixer.generate_mixed_animation(
    prompt="A cute cartoon character dancing in a magical forest",
    temo_strength=0.5,
    felfel_strength=0.5,
    seed=42
)

print(f"Animation saved to: {output_path}")
```

### 3. Using the Enhanced Mixer

```python
from enhanced_animdiff_mixer import EnhancedAnimDiffMixer

# Initialize with config
mixer = EnhancedAnimDiffMixer("config.yaml")

# Generate with preset
output_path = mixer.generate_animation(
    prompt="Two cartoon friends having an adventure",
    preset="balanced_mix",
    seed=42
)
```

## Configuration ⚙️

The `config.yaml` file contains all settings for maximum quality:

### Quality Settings
```yaml
quality:
  num_inference_steps: 50    # Higher = better quality
  guidance_scale: 7.5        # Prompt adherence
  num_frames: 16             # Animation length
  height: 512                # Resolution
  width: 512                 # Resolution
  fps: 8                     # Smoothness
```

### Character Mixing Presets
```yaml
character_mixing:
  presets:
    temo_dominant:      # 80% Temo, 20% Felfel
    balanced_mix:       # 50% Temo, 50% Felfel
    felfel_dominant:    # 20% Temo, 80% Felfel
    creative_blend:     # 60% Temo, 40% Felfel
    artistic_fusion:    # 30% Temo, 70% Felfel
```

## Usage Examples 🎨

### Generate Comparison Set
```python
mixer = AnimDiffCharacterMixer()

# Generate multiple variations
variations = mixer.generate_comparison_animations(
    "A cartoon character exploring a colorful world"
)

for desc, path in variations:
    print(f"{desc}: {path}")
```

### Custom Mixing Ratios
```python
# Heavily favor Temo character
output = mixer.generate_mixed_animation(
    prompt="A brave cartoon hero on an adventure",
    temo_strength=0.8,
    felfel_strength=0.2,
    seed=123
)
```

### Batch Generation
```python
prompts = [
    "A cartoon character in a magical forest",
    "Two friends dancing under the stars",
    "A character flying through clouds"
]

results = mixer.batch_generate(prompts, ["balanced_mix"] * 3)
```

## Advanced Features 🔧

### xFormers Optimization
The system automatically enables xFormers for memory efficiency:
- Memory-efficient attention
- Reduced VRAM usage
- Faster generation times

### Quality Enhancements
- **EulerDiscreteScheduler**: Better sampling quality
- **Attention Slicing**: Memory optimization
- **VAE Slicing**: Reduced memory usage
- **CPU Offloading**: Handle large models

### Prompt Enhancement
Automatic prompt enhancement adds quality keywords:
- "high quality cartoon animation"
- "vibrant colors"
- "smooth motion"
- "detailed character design"

## Troubleshooting 🔧

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_frames` in config
   - Enable CPU offloading
   - Lower resolution temporarily

2. **LoRA Files Not Found**
   - Check file paths in config
   - Ensure `.pt` files exist in character directories

3. **Slow Generation**
   - Reduce `num_inference_steps` for testing
   - Enable all memory optimizations
   - Use lower resolution for tests

### Memory Optimization
```python
# For lower VRAM systems
config['quality']['height'] = 256
config['quality']['width'] = 256
config['quality']['num_frames'] = 8
config['quality']['num_inference_steps'] = 25
```

## Output Examples 🎥

Generated animations will be saved as high-quality GIF files:
- **Format**: GIF (configurable to MP4)
- **Resolution**: 512x512 (configurable)
- **Frame Rate**: 8 FPS (configurable)
- **Duration**: ~2 seconds (16 frames)

## Performance Tips 💡

1. **First Run**: Initial model downloads may take time
2. **Subsequent Runs**: Models are cached for faster loading
3. **Batch Processing**: More efficient than individual generations
4. **Seed Usage**: Use seeds for reproducible results

## File Outputs 📁

```
outputs/
├── animation_0.5_0.5_42.gif          # Balanced mix
├── animation_0.8_0.2_43.gif          # Temo dominant
├── animation_0.2_0.8_44.gif          # Felfel dominant
└── test_outputs/
    └── test_animation.gif             # Test output
```

## License 📄

This project uses the following models:
- **AnimDiff**: guoyww/animatediff-motion-adapter-v1-5-2
- **Base Model**: runwayml/stable-diffusion-v1-5
- **Custom LoRAs**: Your trained Temo and Felfel models

## Support 🆘

If you encounter issues:
1. Run `test_animdiff.py` to diagnose problems
2. Check CUDA availability with `torch.cuda.is_available()`
3. Verify LoRA file integrity
4. Monitor VRAM usage during generation

---

**Happy Animating!** 🎬✨ 