# Adding New Models to MedVision



## Codebase Architecture

```
├── src
	├── medvision_bm 
		├── medvision_lmms_eval
			├── lmms_eval
				├── models
					├── __init__.py # [1]
					├── vllm_qwen25vl.py # [2]
					├── <other-model>
```

- [1] New models registration should be added to the dictionary `AVAILABLE_MODELS`.

  Currently supported models:

  ```python
  AVAILABLE_MODELS = {
      "vllm_gemma3": "VLLM_Gemma3",
      "vllm_internvl3": "VLLM_InternVL3",
      "vllm_llama_3_2_vision": "VLLM_Llama_3_2_Vision",
      "vllm_llava_onevision": "VLLM_Llava_OneVision",
      "vllm_qwen25vl": "VLLM_Qwen25VL",
      "lingshu": "Lingshu",
      "medgemma": "MedGemma",
      "biomedgpt": "BiomedGPT",
      "healthgpt_l14": "HealthGPT_L14",
      "healthgpt_xl32": "HealthGPT_XL32",
      "healthgpt": "HealthGPT",
      "llava_med": "LLaVA_Med",
      "meddr": "MedDr",
      "huatuogpt_vision": "HuatuoGPT_Vision",
      "gemini__2_5_woTool": "Gemini__2_5_woTool",
      "gemini__2_5": "Gemini__2_5",
      "internvl3": "InternVL3",
      "llama_vision": "LlamaVision",
      "llama4": "Llama4",
      "llava_onevision": "Llava_OneVision",
      "qwen2_5_vl": "Qwen2_5_VL",
      "vllm": "VLLM",
  }
  ```

  - `vllm_*` means using vLLM inference engine

- [2] Model file

## Steps

1. Add a model file like `vllm_qwen25vl.py` and add model to `AVAILABLE_MODELS` in `__init__.py`


> [!TIP]
>
> 1. Match model class name and registry name in model file and those in `__init__.py`
> 2. Define `generate_until()` in model file

2. Implement a model-specific image processing function for your model in  [`src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py`](https://github.com/YongchengYAO/MedVision/blob/4593141e3a36da67dd23e64100b51180b6171e04/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/medvision/medvision_utils.py#L418)

   ```python
   def get_resized_img_shape(model_name, img_2d_raw, lmms_eval_specific_kwargs):
       # Supported models
       supported_models = ["qwen2_5_vl", "medgemma", "meddr", "llava_onevision", "llava_med", "llama_3_2_vision", "internvl3", "huatuogpt_vision", "healthgpt_l14", "gemma3", "lingshu"]
       if model_name not in supported_models:
           raise ValueError(f"Model {model_name} is not supported for tumor/lesion size estimation task.\n You need to add model-specific implementation in this function: doc_to_text_TumorLesionSize().")
   
       # Get reshaped image size so that we can adjust the pixel size dynamically
       if model_name == "qwen2_5_vl":
           # NOTE: Qwen2.5-VL resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
           # Preprocessor config: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/blob/main/preprocessor_config.json
           # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
           img_shape_resized_hw = _process_img_qwen25vl(img_2d_raw, lmms_eval_specific_kwargs)
       elif model_name == "lingshu":
           # NOTE: Lingshu resizes images to a size divisible by patch_size (default 14) * merge_size (default 2) = 28
           # Preprocessor config: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B/blob/main/preprocessor_config.json
           # Image processor - Qwen2VLImageProcessor: https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L84
           img_shape_resized_hw = _process_img_lingshu(img_2d_raw, lmms_eval_specific_kwargs)
       elif model_name == "llama_3_2_vision":
           # NOTE: Llama-3.2-Vision dynamically resize the image to a shape that can fit in patches of size [560, 560].
           # Preprocessor config: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/preprocessor_config.json
           # Image processor - MllamaImageProcessor: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py#L536
           img_shape_resized_hw = _process_img_llama_3_2_vision(img_2d_raw, lmms_eval_specific_kwargs)
       elif model_name == "llava_onevision":
           # NOTE: Llava-OneVision dynamically resize the image to a shape that can fit in patches of size [384,384]
           # NOTE: The current probing method only work for single image input, as padding is enabled for multiple image inputs
           # Preprocessor config: https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf/blob/main/preprocessor_config.json
           # Image processor - LlavaOnevisionImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/llava_onevision/image_processing_llava_onevision.py#L108
           img_shape_resized_hw = _process_img_llavaonevision(img_2d_raw, lmms_eval_specific_kwargs)
       elif model_name == "gemma3":
           # NOTE: HealthGPT resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/google/gemma-3-27b-it/blob/main/preprocessor_config.json
           # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
           img_shape_resized_hw = [896, 896]
           # img_shape_resized_hw = _process_img_gemma3(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
       elif model_name == "medgemma":
           # NOTE: Medgemma resize images to a fixed size [896, 896]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/google/medgemma-4b-it/blob/main/preprocessor_config.json
           # Image processor - Gemma3ImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/gemma3/image_processing_gemma3.py#L53
           img_shape_resized_hw = [896, 896]
           # img_shape_resized_hw = _process_img_medgemma(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
       elif model_name == "meddr":
           # NOTE: MedDr resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
           # Check the fixed size in the model config: https://huggingface.co/Sunanhe/MedDr_0401/blob/main/config.json
           img_shape_resized_hw = [448, 448]
           # img_shape_resized_hw = _process_img_meddr(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
       elif model_name == "llava_med":
           # NOTE: Llava-Med resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           # Check the fixed size in the model config: https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b/blob/main/config.json
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_llavamed(img_2d_raw, lmms_eval_specific_kwargs) # for debugging only
       elif model_name == "internvl3":
           # NOTE: InternVL3 resizes images to a fixed size [448, 448]. We used this size for pixel size adjustment.
           # Preprocessor config: https://huggingface.co/OpenGVLab/InternVL3-38B/blob/main/preprocessor_config.json
           # Image processor - CLIPImageProcessor: https://github.com/huggingface/transformers/blob/91393fe4cc3266a05bc0d129e34ff5f761bb46e2/src/transformers/models/clip/image_processing_clip.py#L54
           img_shape_resized_hw = [448, 448]
           # img_shape_resized_hw = _process_img_internvl3(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
       elif model_name == "huatuogpt_vision":
           # NOTE: HuatuoGPT-Vision resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           # The fixed size is configured in the "shortest_edge" in image processor: https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-34B-hf/blob/main/preprocessor_config.json
           # Image processor - CLIPImageProcessor:
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_huatuogpt_vision(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
       elif model_name == "healthgpt_l14":
           # NOTE: HealthGPT resize images to a fixed size [336, 336]. We used this size for pixel size adjustment.
           img_shape_resized_hw = [336, 336]
           # img_shape_resized_hw = _process_img_healthgpt_L14(img_2d_raw, lmms_eval_specific_kwargs)  # for debugging only
       return img_shape_resized_hw
   ```

   > [!TIP]
   >
   > **Function**:
   >
   > This function is used to align the physical spacing information (i.e., pixel size) in the text prompt with the images perceived by the model. 
   >
   > **Motivation**:
   >
   > Each VLM has their own image processor, which has very different behavior –  while some resize images to a predefined fixed size, the other may adopt a dynamic resize (“smart resize”) strategy. Since our task is scale-sensitive, we need to assure that the pixel and image size in the prompt is correct. If you just read the image and pixel size from the original image, and put this info in the prompt, it could mislead the model since the actual input image has been resized internally.
   >
   > **Strategy**:
   >
   > - For model with fixed input size, set the input image size
   > - For dynamic processing model, use the image processor to process each image and get the new image size. For this purpose, you need to set a `sample_model_hf` for loading the image processor. See example in the next step.

3. Set `sample_model_hf` for models with dynamic image processing scheme in the base task yaml files, such as [src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/MSD/MSD_BoxCoordinate_base.yaml](https://github.com/YongchengYAO/MedVision/blob/4593141e3a36da67dd23e64100b51180b6171e04/src/medvision_bm/medvision_lmms_eval/lmms_eval/tasks/MSD/MSD_BoxCoordinate_base.yaml)

   ```yaml
   lmms_eval_specific_kwargs:
     medgemma:
     biomedgpt:
     healthgpt:
     llava_med:
     meddr:
     huatuogpt_vision:
     gemini__2_5:
     gemini__2_5_woTool:
     internvl3:
     llava_onevision:
     qwen2_5_vl:
       model: "qwen2_5_vl"
       sample_model_hf: "Qwen/Qwen2.5-VL-32B-Instruct"
     vllm_qwen25vl:
       model: "qwen2_5_vl"
       sample_model_hf: "Qwen/Qwen2.5-VL-32B-Instruct"
   ```

   > [!TIP]
   >
   > - `model`: the model name in `AVAILABLE_MODELS`
   > - `sample_model_hf`: the HF model ID of an example model to load the image processor

## Reference

- [New models guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/model_guide.md) from `EvolvingLMMs-Lab/lmms-eval `