from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, load_in_4bit: bool=False) -> ModelBase:

    if 'qwen3.5' in model_path.lower():
        from pipeline.model_utils.qwen35_model import Qwen35Model
        return Qwen35Model(model_path, load_in_4bit=load_in_4bit)
    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path) 
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
