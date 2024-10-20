# ComfyUI GLM-4 Wrapper

This repository contains custom nodes for ComfyUI, specifically designed to enhance and infer prompts using the GLM-4 model on local hardware.

The nodes leverage the GLM-4 model to generate detailed and descriptive image/video captions or enhance user-provided prompts, among regular inference.

Prompts and inference can be combined with image if `THUDM/glm-4v-9b` model is used.

All models will be downloaded automatically through HuggingFace.co. `THUDM/glm-4v-9b` will hold ~26 GB of hdd space and `THUDM/glm-4-9b` will hold ~18 GB of hdd space.

The nodes containes an "unload_model" option which frees up VRAM space and makes it suitable for workflows that requires larger VRAM space, like FLUX.1-dev and CogVideoX-5b(-I2V).

The prompt enhancer is based on this example from THUDM [convert_demo.py](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py).
Thier demo is only for usage through OpenAI API and I wanted to build something local.

Hope you will enjoy your enhanced prompts and inference capabilities of these models. They are great!

## Update 2024-10-03

Added support for quantized models. They are performing exceptionally well. Check metrics below.

Model `alexwww94/glm-4v-9b-gptq-4bit` is significatly more lightweight than the original and will hold ~8.5 GB of hdd space.

Model `alexwww94/glm-4v-9b-gptq-3bit` is even more lightweight and will hold ~7.6 GB of hdd space.

[Link to metrics](https://huggingface.co/alexwww94/glm-4v-9b-gptq-4bit#metrics)


## Features

- **GLM-4 Model Loader**: Load various GLM-4 models with different precision and quantization settings.
- **GLM-4 Prompt Enhancer**: Enhances base prompts using the GLM-4 model.
- **GLM-4 Inferencing**: Performs inference using various GLM-4 models.

## Installation

1. Navigate to ComfyUI custom nodes:
  ```bash
  cd /your/path/to/ComfyUI/ComfyUI/custom_nodes/
  ```
2. Clone the repository:
  ```bash
  git clone https://github.com/Nojahhh/ComfyUI_GLM4_Wrapper.git
  ```
3. Navigate to the cloned directory:
  ```bash
  cd ComfyUI_GLM4_Wrapper
  ```
4. Install the required dependencies:
  ```bash
  ../../python_embeded python.exe -m pip install -r requirements.txt
  ```

## Usage

### GLM-4 Model Loader

The `GLM4ModelLoader` class is responsible for loading GLM-4 models. It supports various models and precision settings.

#### Input Types

- **model**: Choose the GLM-4 model to load. Model will download automatically from HuggingFace.co
- **precision**: Precision type (`fp16`, `fp32`, `bf16`).
`THUDM/glm-4v-9b` requires `bf16` and is set to run in 4-bit by default based on it's size.
`alexwww94/glm-4v-9b-gptq-4bit` requires `bf16` and is set to run in 4-bit by default.
`alexwww94/glm-4v-9b-gptq-3bit` requires `bf16` and is set to run in 3-bit by default.
- **quantization**: Set the number of bits for quantization (`4`, `8`, `16`). Default value of `4`. (This option is bypassed when using the GPTQ-models).

#### Output

- **GLM4Pipeline**: The GLM-4 pipeline.

### GLM-4 Prompt Enhancer

Enhances a given prompt using the GLM-4 model.

#### Input Parameters

- **GLMPipeline**: Provide a GLM-4 pipeline.
- **prompt**: Base prompt to enhance.
- **max_tokens**: Maximum number of output tokens.
- **temperature**: Temperature parameter for sampling.
- **top_k**: Top-k parameter for sampling.
- **top_p**: Top-p parameter for sampling.
- **repetition_penalty**: Repetition penalty for sampling.
- **image** (optional): Image to enhance the prompt. Only works with `THUDM/glm-4v-9b`, `alexwww94/glm-4v-9b-gptq-4bit` and `alexwww94/glm-4v-9b-gptq-3bit`.
- **unload_model**: Unload the model after use.

#### Output

- **enhanced_prompt**: The enhanced prompt.

### GLM-4 Inferencing

Performs inference using the GLM-4 model.

#### Input Parameters

- **GLMPipeline**: Provide a GLM-4 pipeline.
- **system_prompt**: System prompt for inferencing.
- **user_prompt**: User prompt for inferencing.
- **max_tokens**: Maximum number of output tokens.
- **temperature**: Temperature parameter for sampling.
- **top_k**: Top-k parameter for sampling.
- **top_p**: Top-p parameter for sampling.
- **repetition_penalty**: Repetition penalty for sampling.
- **image** (optional): Image to use as input for inferencing. Only works with `THUDM/glm-4v-9b`, `alexwww94/glm-4v-9b-gptq-4bit` and `alexwww94/glm-4v-9b-gptq-3bit`.
- **unload_model**: Unload the model after use.

#### Output

- **output_text**: The generated text from the model.

## Node Class Mappings

- **GLM-4 Model Loader**: `GLM4ModelLoader`
- **GLM-4 Prompt Enhancer**: `GLM4PromptEnhancer`
- **GLM-4 Inferencing**: `GLM4Inference`

## Node Display Name Mappings

- **GLM-4ModelLoader**: "GLM-4 Model Loader"
- **GLM-4PromptEnhancer**: "GLM-4 Prompt Enhancer"
- **GLM-4Inference**: "GLM-4 Inferencing"

## Supported Models

The following GLM-4 models are supported by this wrapper:

| Model Name                      | Size  | Recommended Precision  |
|---------------------------------|-------|------------------------|
| `alexwww94/glm-4v-9b-gptq-4bit` | 9B    | `bf16` (4-bit quant)   |
| `alexwww94/glm-4v-9b-gptq-3bit` | 9B    | `bf16` (3-bit quant)   |
| `THUDM/glm-4v-9b`               | 9B    | `bf16` (4/8-bit quant) |
| `THUDM/glm-4-9b`                | 9B    | `fp16`, `fp32`, `bf16` |
| `THUDM/glm-4-9b-chat`           | 9B    | `fp16`, `fp32`, `bf16` |
| `THUDM/glm-4-9b-chat-1m`        | 9B    | `fp16`, `fp32`, `bf16` |
| `THUDM/LongCite-glm4-9b`        | 9B    | `fp16`, `fp32`, `bf16` |
| `THUDM/LongWriter-glm4-9b`      | 9B    | `fp16`, `fp32`, `bf16` |

### Notes:
- `THUDM/glm-4v-9b` requires `bf16` precision and is default 4-bit quantization due to its size and the typical VRAM limitations of consumer-grade GPUs (often 24GB or less).
- `alexwww94/glm-4v-9b-gptq-4bit` requires `bf16` and is default 4-bit.
- `alexwww94/glm-4v-9b-gptq-3bit` requires `bf16` and is default 3-bit.
- Only `THUDM/glm-4v-9b`, `alexwww94/glm-4v-9b-gptq-4bit` and `alexwww94/glm-4v-9b-gptq-3bit` models are able to handle image input.

## Example Usage

Below is an example of how to use the GLM-4 Prompt Enhancer and GLM-4 Inferencing nodes in your code:

### GLM-4 Prompt Enhancer

```python
from comfyui_glm4_wrapper import GLM4ModelLoader, GLM4PromptEnhancer, GLM4Inference

# Load the model
model_loader = GLM4ModelLoader()
pipeline = model_loader.gen(model="THUDM/glm-4v-9b", precision="bf16", quantization="8")[0]

# Enhance the prompt
enhancer = GLM4PromptEnhancer()
enhanced_prompt = enhancer.enhance_prompt(
  GLMPipeline=pipeline,
  prompt="A beautiful sunrise over the mountains",
  max_tokens=200,
  temperature=0.1,
  top_k=40,
  top_p=0.7,
  repetition_penalty=1.1,
  image=None,  # PIL Image
  unload_model=True
)
print(enhanced_prompt)
```

### GLM-4 Inferencing

```python
from comfyui_glm4_wrapper import GLM4ModelLoader, GLM4PromptEnhancer, GLM4Inference

# Load the model
model_loader = GLM4ModelLoader()
pipeline = model_loader.gen(model="THUDM/glm-4v-9b", precision="bf16", quantization="8")[0]

# Perform inference
inference = GLM4Inference()
output_text = inference.infer(
  GLMPipeline=pipeline,
  system_prompt="Describe the scene in detail:",
  user_prompt="A bustling city street at night",
  max_tokens=250,
  temperature=0.7,
  top_k=50,
  top_p=1,
  repetition_penalty=1.0,
  image=None,
  unload_model=True
)
print(output_text)
```

For more detailed examples and advanced usage, please refer to the documentation or the example scripts provided in the repository.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [ComfyUI](https://github.com/comfyui)
- [Transformers](https://github.com/huggingface/transformers)
- [THUDM](https://github.com/THUDM)

## Contact

For any questions or feedback, please open an issue on GitHub or contact me at [mellin.johan@gmail.com](mailto:mellin.johan@gmail.com).
