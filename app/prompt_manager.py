import yaml
from string import Template
from functools import lru_cache
class PromptManager:
    def __init__(self, config_path="./prompts/intent_detection.yaml"):
        self.config_path = config_path
        self.prompts = self._load_yaml()

    @staticmethod
    def _load_yaml_from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @lru_cache()
    def _load_yaml(self):
        data = self._load_yaml_from_file(self.config_path)
        if "prompts" not in data:
            raise ValueError("prompt.yaml 必须包含 'prompts' 字段")
        return data["prompts"]

    def get_roles(self):
        """返回所有角色名"""
        return list(self.prompts.keys())

    def get_prompt(self, role):
        """返回指定角色的 prompt 内容（不渲染）"""
        if role not in self.prompts:
            raise KeyError(f"未找到角色 prompt: {role}")
        return self.prompts[role]

    def render(self, role, **kwargs):
        """
        渲染某个角色的 prompt.
        支持 {{var}} 风格（自动转换为 $var）
        """
        if role not in self.prompts:
            raise KeyError(f"未找到角色 prompt: {role}")

        item = self.prompts[role]

        system_text = item["system"]
        user_text = item["user_prompt"]
        # 支持 {{var}} -> $var 的轻量转换
        for k in kwargs:
            user_text = user_text.replace(f"{{{{{k}}}}}", f"${k}")

        system_template = Template(system_text)
        user_template = Template(user_text)
        
        system_rendered_text = system_template.safe_substitute(**kwargs)
        user_rendered_text = user_template.safe_substitute(**kwargs)

        return {
            "system": system_rendered_text,
            "user": user_rendered_text
        }

    def reload(self):
        """重新加载 YAML（适合热更新）"""
        self._load_yaml.cache_clear()
        self.prompts = self._load_yaml()
