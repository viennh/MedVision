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

## Keys

1. Match model class name and registry name in model file and those in `__init__.py`
2. Define `generate_until()` in model file

## Reference

- [New models guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/model_guide.md) from `EvolvingLMMs-Lab/lmms-eval `