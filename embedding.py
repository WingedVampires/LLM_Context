import traceback
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

class Embedding:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            print("创建新的 Embedding 实例")
            cls._instance = super().__new__(cls)
            
            try:
                print("初始化 pipeline...")
                # 动态检测设备：优先 MPS (Apple Silicon)，否则 CPU
                # if torch.backends.mps.is_available():
                #     cls._device = "mps"
                #     print("Using MPS device for acceleration on Apple Silicon.")
                # else:
                #     cls._device = "cpu"
                #     print("Using CPU device (no MPS/CUDA available).")
                model_path = "/root/.cache/huggingface/hub/models--jinaai--jina-embeddings-v2-base-code/snapshots/516f4baf13dec4ddddda8631e019b5737c8bc250"
                cls._device = 'cuda'
                # 加载模型和 tokenizer 到检测的设备
                cls._model = AutoModel.from_pretrained(
                    model_path, 
                    # trust_remote_code=True
                ).to(cls._device)
                cls._tokenizer = AutoTokenizer.from_pretrained(model_path)
                cls._model.eval()  # 设置为评估模式
                print("embedding model 初始化成功")
            except Exception as e:
                print(f"pipeline 初始化失败: {e}")
                raise
        return cls._instance
    
    def __init__(self):
        pass
    
    def get_embedding(self, text):
        """获取文本的 embedding"""
        try:
            if text is None:
                print("警告: 输入文本为 None")
                return None
                
            if not isinstance(text, str):
                print(f"警告: 输入文本类型不是字符串，而是 {type(text)}")
                text = str(text)
                
            if not text.strip():
                print("警告: 输入文本为空")
                return None
            
            # Tokenize 输入
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # 移动到模型设备
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self._model(**inputs)
                # 使用最后一个隐藏状态的 mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
            
            return embedding
            
        except Exception as e:
            print(f"获取 embedding 时出错: {e}")
            print(f"model 状态: {self._model}")
            print(traceback.format_exc())
            return None

    def _cos_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def text_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        vec1 = self.get_embedding(text1)
        vec2 = self.get_embedding(text2)
        if vec1 is None or vec2 is None:
            return 0.0
        return self._cos_similarity(vec1, vec2)
