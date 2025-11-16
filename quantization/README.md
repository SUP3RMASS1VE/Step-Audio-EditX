# Model Quantization Scripts

This directory contains multiple quantization scripts for the Step-Audio-EditX model. Due to the custom architecture, different quantization methods have varying levels of compatibility.

## Available Methods

### 1. BitsAndBytes (Recommended - Most Compatible)

**Script:** `bnb_quantize.py`

The most reliable method for custom architectures. Uses bitsandbytes library for 4-bit or 8-bit quantization.

```bash
# 4-bit quantization (recommended)
python quantization/bnb_quantize.py --model_path models/Step-Audio-EditX/Step-Audio-EditX --bits 4

# 8-bit quantization (higher quality, larger size)
python quantization/bnb_quantize.py --model_path models/Step-Audio-EditX/Step-Audio-EditX --bits 8
```

**Pros:**
- Works reliably with custom model architectures
- Easy to use
- Good compression ratio
- Fast loading

**Cons:**
- Requires bitsandbytes library
- Slightly lower quality than GPTQ/AWQ

### 2. GPTQ (Alternative)

**Script:** `gptq_quantize.py`

Uses AutoGPTQ for quantization. May have compatibility issues with custom architectures.

```bash
python quantization/gptq_quantize.py --model_path models/Step-Audio-EditX/Step-Audio-EditX
```

**Pros:**
- Better quality than bitsandbytes
- Good compression

**Cons:**
- May not work with all custom architectures
- Requires calibration dataset
- Slower quantization process

### 3. AWQ (Advanced - May Have Issues)

**Script:** `awq_quantize.py`

Uses llm-compressor for AWQ quantization. Currently has compatibility issues with the Step1 architecture due to torch.fx tracing limitations.

```bash
python quantization/awq_quantize.py --model_path models/Step-Audio-EditX/Step-Audio-EditX
```

**Known Issues:**
- torch.fx cannot trace custom Step1 architecture
- Fails during AWQ calibration phase
- Not recommended for this model

## Installation

Install required packages:

```bash
# For BitsAndBytes (recommended)
pip install bitsandbytes accelerate

# For GPTQ
pip install auto-gptq optimum

# For AWQ (currently not working)
pip install llmcompressor
```

## Recommendation

**Use BitsAndBytes** (`bnb_quantize.py`) for the Step-Audio-EditX model. It's the most compatible with custom architectures and provides good results.

## Loading Quantized Models

### BitsAndBytes:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "models/Step-Audio-EditX/Step-Audio-EditX/bnb-4bit",
    quantization_config=config,
    trust_remote_code=True,
    device_map="auto"
)
```

### GPTQ:
```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "models/Step-Audio-EditX/Step-Audio-EditX/gptq-4bit",
    trust_remote_code=True,
    device_map="auto"
)
```

## Troubleshooting

### AWQ torch.fx tracing errors
The AWQ method uses torch.fx to trace the model, which doesn't work well with custom architectures that have:
- Custom attention mechanisms
- Dynamic control flow
- Custom operations

**Solution:** Use BitsAndBytes instead.

### Out of memory errors
- Reduce `num_calibration_samples`
- Use 8-bit instead of 4-bit
- Ensure no other processes are using GPU memory

### Model loading errors
- Ensure `trust_remote_code=True` is set
- Check that all model files are present
- Verify CUDA is available for GPU quantization
