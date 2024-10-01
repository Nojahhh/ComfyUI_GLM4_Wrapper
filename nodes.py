# This file contains the node classes for the GLM-4 wrapper nodes in the Comfy platform.
# The GLM-4 wrapper nodes are used to interact with the GLM-4 models for enhancing prompts and inferencing.
# The GLM-4 models are used for text generation tasks and image to video captioning tasks.
# Author: Johan Mellin, 2024, Stockholm, Sweden

import torch
import comfy.model_management as mm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import logging
import numpy as np
import gc

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class GLMPipeline:
  def __init__(self):
    self.tokenizer = None
    self.transformer = None
    self.model_name = None
    self.precision = None
    self.quantization = None
    self.parent = None

  def clearCache(self):
    torch.cuda.synchronize()
    if self.transformer:
      self.transformer.cpu()
      del self.transformer
    self.tokenizer = None
    self.transformer = None
    self.model_name = None
    self.precision = None
    self.quantization = None

class GLM4ModelLoader:

  def __init__(self):
    self.model = None
    self.precision = None
    self.quantization = None
    self.pipeline = GLMPipeline()
    self.pipeline.parent = self
    pass

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "model": (
          [
            "THUDM/glm-4v-9b",
            "THUDM/glm-4-9b",
            "THUDM/glm-4-9b-chat",
            "THUDM/glm-4-9b-chat-1m",
            "THUDM/LongCite-glm4-9b",
            "THUDM/LongWriter-glm4-9b"
          ],
          {"tooltip": "Choose the GLM-4 model to load. Only glm-4v-9b model supports image input."}
        ),
        "precision": (["fp16", "fp32", "bf16"],
          {"default": "bf16", "tooltip": "Recommended precision for GLM-4 model. bf16 required for glm-4v-9b (INT4/8 quant)."}),
        "quantization": (["4", "8", "16"], {"default": "8", "tooltip": "Choose the number of bits for quantization. Only supported for glm-4v-9b model."}),
      }
    }

  CATEGORY = "GLM4Wrapper"
  RETURN_TYPES = ("GLMPipeline",)
  FUNCTION = "gen"

  def loadCheckPoint(self):
    # Initialize the device and empty cache
    device = mm.get_torch_device()
    mm.soft_empty_cache()

    # Clear cache
    if self.pipeline != None:
      self.pipeline.clearCache()

    # Set precision type
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]

    log.info(f"Loading GLM-4 model: {self.model}")

    # Load the tokenizer and model with specified precision, and trust remote code
    tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

    if(self.model == "THUDM/glm-4v-9b"):
      # Load the model with low_cpu_mem_usage and trust_remote_code
      if(self.quantization == "8"):
        log.info(f"Loading GLM-4 model in 8-bit quantization mode")
        transformer = AutoModelForCausalLM.from_pretrained(self.model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
      elif(self.quantization == "4"):
        log.info(f"Loading GLM-4 model in 4-bit quantization mode")
        transformer = AutoModelForCausalLM.from_pretrained(self.model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(load_in_4bit=True))
      else:
        log.info(f"Loading GLM-4 model in default mode")
        transformer = AutoModelForCausalLM.from_pretrained(self.model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    else:
      transformer = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", trust_remote_code=True).to(dtype).to(device)
    transformer.eval()

    self.pipeline.tokenizer = tokenizer
    self.pipeline.transformer = transformer
    self.pipeline.model_name = self.model
  
  def clearCache(self):
    # if self.pipeline != None:
    self.pipeline.clearCache()
    # mm.soft_empty_cache()

  def gen(self,model,precision,quantization):
    if self.model == None or self.model != model or self.pipeline == None:
      self.model = model
      self.precision = precision
      self.quantization = quantization
      self.loadCheckPoint()
    return (self.pipeline,)

# Class for enhancing prompts using GLM-4
class GLM4PromptEnhancer:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "GLMPipeline": ("GLMPipeline", {"tooltip": "Provide a GLM-4 pipeline."}),
        "prompt": ("STRING", {"forceInput": True, "tooltip": "Provide a base prompt to enhance. Can be empty if image is provided and glm-4v-9b model is chosen."}),
        "max_tokens": ("INT", {"default": 200, "tooltip": "Limit the number of output tokens"}),
        "temperature": ("FLOAT", {"default": 0.1, "tooltip": "Temperature parameter for sampling"}),
        "top_k": ("INT", {"default": 40, "tooltip": "Top-k parameter for sampling"}),
        "top_p": ("FLOAT", {"default": 0.7, "tooltip": "Top-p parameter for sampling"}),
        "repetition_penalty": ("FLOAT", {"default": 1.1, "tooltip": "Repetition penalty for sampling"}),
        "unload_model": ("BOOLEAN", {"default": True, "tooltip": "Unload the model after use to free up memory"}),
      },
      "optional": {
        "image": ("IMAGE", {"tooltip": "Provide an image to enhance the prompt. Only supported for glm-4v-9b model."}),
      }
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("enhanced_prompt",)
  FUNCTION = "enhance_prompt"
  CATEGORY = "GLM4Wrapper"

  def enhance_prompt(self, GLMPipeline, prompt, max_tokens=200, temperature=0.1, top_k=40, top_p=0.7, repetition_penalty=1.1, image=None, unload_model=True):
    # Empty cache
    mm.soft_empty_cache()

    # Load the model if it is not loaded
    if GLMPipeline.tokenizer == None :
      GLMPipeline.parent.loadCheckPoint()

    # Write the system prompt for enhancing the prompt
    sys_prompt_t2v = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
    There are a few rules to follow:

    You will only ever output a single video description per user request.

    When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
    Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

    Video descriptions must have the same num of words as examples below. Extra words will be ignored.
    """

    # Write the system prompt for image to video captioning
    sys_prompt_i2v = """
    **Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input. 

    **Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. User input is optional and can be empty. 

    **Note**: Be assertive and confident in your descriptions. Don't fall back to "perhaps" or "maybe". Avoid "suggests" or "implies" unless it's a clear-cut case.

    **Note**: Don't use too rapid or too slow motion. Keep the motion at a moderate pace.

    **Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

    **Answering Style**:
    Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more. For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

    **Output Format**: "[highly descriptive image caption here]"

    user input:
    """

    # Check if the model is GLM-4v-9b for image to video captioning
    if(GLMPipeline.model_name == "THUDM/glm-4v-9b"):

      # Add an explicit instruction to enhance the prompt
      if image is not None:
        image = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")
        messages=[{"role": "user", "image": image, "content": f"{sys_prompt_i2v} {prompt}"}]
      else:
        messages=[{"role": "user", "content": f"{sys_prompt_t2v} {prompt}"}]

      # Tokenize the input text with the instruction
      inputs = GLMPipeline.tokenizer.apply_chat_template(messages,
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
        return_dict=True)

    else:

      # Add an explicit instruction to enhance the prompt
      messages=[
        {"role": "system", "content": f"{sys_prompt_t2v}"},
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
        },
        {
            "role": "assistant",
            "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
        },
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
        },
        {
            "role": "assistant",
            "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
        },
        {
            "role": "user",
            "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
        },
        {
            "role": "assistant",
            "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
        },
        {
            "role": "user",
            "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {prompt} "',
        },
      ]

      # Tokenize the input text with the instruction
      inputs = GLMPipeline.tokenizer.apply_chat_template(messages, 
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True)

    # Move inputs to the same device as the transformer
    inputs = {key: value.to(GLMPipeline.transformer.device) for key, value in inputs.items()}

    # Generate enhanced text
    with torch.no_grad():
      GLMPipeline.transformer.eval()
      outputs = GLMPipeline.transformer.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
      enhanced_text = GLMPipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the system prompt from the output text
    for message in messages:
      enhanced_text = enhanced_text.replace(message["content"], "").strip()

    # Clean up the enhanced text
    if enhanced_text.startswith('"'):
      enhanced_text = enhanced_text[1:]
    if enhanced_text.endswith('"'):
      enhanced_text = enhanced_text[:-1]
    if enhanced_text.startswith('[') and "]" in enhanced_text:
      enhanced_text = enhanced_text.split("]")[0]
    if enhanced_text.startswith('['):
      enhanced_text = enhanced_text[1:]
    if enhanced_text.endswith(']'):
      enhanced_text = enhanced_text[:-1]
    enhanced_text = enhanced_text.replace('Captivating scene:', '').strip()

    # Remove any extra newlines or carriage returns
    if "\r" in enhanced_text:
      enhanced_text = enhanced_text.split("\r")[0]
    if "\n" in enhanced_text:
      enhanced_text = enhanced_text.split("\n")[0]

    if unload_model == True:
      GLMPipeline.parent.clearCache()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      gc.collect()
    
    return (enhanced_text,)
  
# GLM-4 Inferencing node
class GLM4Inference:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "GLMPipeline": ("GLMPipeline", {"tooltip": "Provide a GLM-4 pipeline."}),
        "system_prompt": ("STRING", {"default":"", "multiline": True, "tooltip": "Provide a system prompt for inferencing. (Instructions for the model)"}),
        "user_prompt": ("STRING", {"default":"", "multiline": True, "tooltip": "Provide a user prompt for inferencing"}),
        "max_tokens": ("INT", {"default": 250, "tooltip": "Limit the number of output tokens"}),
        "temperature": ("FLOAT", {"default": 0.7, "tooltip": "Temperature parameter for sampling"}),
        "top_k": ("INT", {"default": 50, "tooltip": "Top-k parameter for sampling"}),
        "top_p": ("FLOAT", {"default": 1, "tooltip": "Top-p parameter for sampling"}),
        "repetition_penalty": ("FLOAT", {"default": 1.0, "tooltip": "Repetition penalty for sampling"}),
      },
      "optional": {
        "image": ("IMAGE", {"tooltip": "Provide an image to use as input for inferencing. Only supported for glm-4v-9b model."}),
        "unload_model": ("BOOLEAN", {"default": True, "tooltip": "Unload the model after use to free up memory"}),
      }
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("output_text",)
  FUNCTION = "infer"
  CATEGORY = "GLM4Wrapper"

  def infer(self, GLMPipeline, system_prompt, user_prompt, max_tokens=250, temperature=0.7, top_k=50, top_p=1, repetition_penalty=1.0, image=None, unload_model=True):
    # Empty cache
    mm.soft_empty_cache()

    # Load the model if it is not loaded
    if GLMPipeline.tokenizer == None :
      GLMPipeline.parent.loadCheckPoint()

    # # Check if the model is GLM-4v-9b for image to video captioning
    if GLMPipeline.model_name == "THUDM/glm-4v-9b":

      # Add an explicit instruction to enhance the prompt
      if image is not None:
        image = Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)).convert("RGB")
        messages = [{"role": "user", "image": image, "content": f"{system_prompt} {user_prompt}"}]
      else:
        messages = [{"role": "user", "content": f"{system_prompt} {user_prompt}"}]

      # Tokenize the input text with the instruction
      inputs = GLMPipeline.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    else:

      # Add an explicit instruction to enhance the prompt
      messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
      ]
      # Tokenize the input text with the instruction
      inputs = GLMPipeline.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)

    # Move inputs to the same device as the transformer
    inputs = {key: value.to(GLMPipeline.transformer.device) for key, value in inputs.items()}

    # Generate enhanced text
    with torch.no_grad():
      GLMPipeline.transformer.eval()
      outputs = GLMPipeline.transformer.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
      output_text = GLMPipeline.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the system prompt from the output text
    for message in messages:
      output_text = output_text.replace(message["content"], "").strip()

    if unload_model == True:
      GLMPipeline.parent.clearCache()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      gc.collect()

    return (output_text,)


# Node class mappings and display names
NODE_CLASS_MAPPINGS = {
  "GLM-4 Model Loader": GLM4ModelLoader,
  "GLM-4 Prompt Enhancer": GLM4PromptEnhancer,
  "GLM-4 Inferencing": GLM4Inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "GLM-4ModelLoader": "GLM-4 Model Loader",
  "GLM-4PromptEnhancer": "GLM-4 Prompt Enhancer",
  "GLM-4Inference": "GLM-4 Inferencing",
}
