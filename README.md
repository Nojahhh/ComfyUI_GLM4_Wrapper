# ComfyUI GLM-4 Wrapper

This repository contains custom nodes for ComfyUI, specifically designed to enhance and infer prompts using the GLM-4 model on local hardware.

The nodes leverage the GLM-4 model to generate detailed and descriptive image/video captions or enhance user-provided prompts, among regular inference.

Prompts and inference can be combined with image if `THUDM/glm-4v-9b` model is used.

All models will be downloaded automatically through HuggingFace.co. `THUDM/glm-4v-9b` will take ~26 GB of hdd space and `THUDM/glm-4-9b` will take ~18 GB of hdd space.

The nodes containes an "unload_model" option which frees up VRAM space and makes it suitable for workflows that requires larger VRAM space, like FLUX.1-dev and CogVideoX-5b(-I2V).

The prompt enhancer is based on this example from THUDM [convert_demo.py](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py).
Thier demo is only for usage through OpenAI API and I wanted to build something local.

Hope you will enjoy your enhanced prompts and inference capabilities of these models. They are great!


## Features

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
  pip install -r requirements.txt
  ```

## Usage

### GLM-4 Prompt Enhancer

Enhances a given prompt using the GLM-4 model.

#### Input Parameters

- **model**: Choose from available GLM-4 models. Model will download automatically from HuggingFace.co.
- **precision**: Precision type (`fp16`, `fp32`, `bf16`). `THUDM/glm-4v-9b` requires bf16 and will be runned in 4-bit quant by default.
- **prompt**: Base prompt to enhance.
- **max_tokens**: Maximum number of output tokens.
- **temperature**: Temperature parameter for sampling.
- **top_k**: Top-k parameter for sampling.
- **top_p**: Top-p parameter for sampling.
- **repetition_penalty**: Repetition penalty for sampling.
- **image** (optional): Image to enhance the prompt. Only works with `THUDM/glm-4v-9b`.
- **unload_model**: Unload the model after use.

#### Output

- **enhanced_prompt**: The enhanced prompt.

### GLM-4 Inferencing

Performs inference using the GLM-4 model.

#### Input Parameters

- **model**: Choose from available GLM-4 models.
- **precision**: Precision type (`fp16`, `fp32`, `bf16`). `THUDM/glm-4v-9b` requires bf16 and will be runned in 4-bit quant by default.
- **system_prompt**: System prompt for inferencing.
- **user_prompt**: User prompt for inferencing.
- **max_tokens**: Maximum number of output tokens.
- **temperature**: Temperature parameter for sampling.
- **top_k**: Top-k parameter for sampling.
- **top_p**: Top-p parameter for sampling.
- **repetition_penalty**: Repetition penalty for sampling.
- **image** (optional): Image to use as input for inferencing. Only works with `THUDM/glm-4v-9b`.
- **unload_model**: Unload the model after use.

#### Output

- **output_text**: The generated text from the model.

## Node Class Mappings

- **GLM-4 Prompt Enhancer**: `GLM4PromptEnhancer`
- **GLM-4 Inferencing**: `GLM4Inference`

## Node Display Name Mappings

- **GLM-4PromptEnhancer**: "GLM-4 Prompt Enhancer"
- **GLM-4Inference**: "GLM-4 Inferencing"

## Supported Models

The following GLM-4 models are supported by this wrapper:

| Model Name                  | Size  | Recommended Precision |
|-----------------------------|-------|-----------------------|
| `THUDM/glm-4v-9b`           | 9B    | `bf16` (INT4 quant)   |
| `THUDM/glm-4-9b`            | 9B    | `fp16`, `fp32`, `bf16`|
| `THUDM/glm-4-9b-chat`       | 9B    | `fp16`, `fp32`, `bf16`|
| `THUDM/glm-4-9b-chat-1m`    | 9B    | `fp16`, `fp32`, `bf16`|
| `THUDM/LongCite-glm4-9b`    | 9B    | `fp16`, `fp32`, `bf16`|
| `THUDM/LongWriter-glm4-9b`  | 9B    | `fp16`, `fp32`, `bf16`|

### Notes:
- The `THUDM/glm-4v-9b` model requires `bf16` precision with INT4 quantization due to its size and the typical VRAM limitations of consumer-grade GPUs (often 24GB or less).
- Only `THUDM/glm-4v-9b` model is able to handle image input.

## Example Usage

Below is an example of how to use the GLM-4 Prompt Enhancer and GLM-4 Inferencing nodes in your code:

### GLM-4 Prompt Enhancer

```python
from comfyui_glm4_wrapper import GLM4PromptEnhancer

enhancer = GLM4PromptEnhancer()
enhanced_prompt = enhancer.enhance_prompt(
  model="THUDM/glm-4v-9b",
  precision="bf16",
  prompt="A beautiful sunrise over the mountains",
  max_tokens=200,
  temperature=0.1,
  top_k=40,
  top_p=0.7,
  repetition_penalty=1.1,
  image=None, # PIL Image
  unload_model=True
)
print(enhanced_prompt)
```

### GLM-4 Inferencing

```python
from comfyui_glm4_wrapper import GLM4Inference

inference = GLM4Inference()
output_text = inference.infer(
  model="THUDM/glm-4-9b",
  precision="fp16",
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
