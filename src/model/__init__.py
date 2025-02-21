from src.model.llama_llm import LLM
from src.model.llama_pt_llm import PromptTuningLLM
from src.model.llama_graph_llm import GraphLLM # use it for LLM finetuning


load_model = {
    "llm": LLM,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "graph_llm": GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "/home/vislab/Yanjia/meta-llama/Llama-2-7b-hf",
    "7b_chat": "/home/vislab/Yanjia/meta-llama/Llama-2-7b-chat-hf",
    "13b": "/home/vislab/Yanjia/meta-llama/Llama-2-13b-hf",
    "13b_chat": "/home/vislab/Yanjia/meta-llama/Llama-2-13b-chat-hf",
}
