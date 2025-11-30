import requests
from .prompt_manager import PromptManager
import json
BASE = "http://hyl_ollama:11434/api/chat"
MODEL = "gpt-oss:20b"

def ask_ollama(prompts):
    payload = {
        "model": MODEL,
        "messages": prompts,
        "max_tokens": 200,
        "stream": False
    }
    r = requests.post(f"{BASE}", json=payload)
    res=r.json().get("message", {}).get("content", "").strip()
    return res

def subquestion_split(query):
    pm = PromptManager()
    query_decomposer = pm.render(
        "query_decomposer",
        query=query
    )
    query_decomposer_messages=[{"role": "system", "content": query_decomposer["system"]},
                {"role": "user", "content": query_decomposer["user"]}]
    query_decomposer_output = ask_ollama(
        prompts=query_decomposer_messages
    )
    return query_decomposer_output



def intent_detection(query, rewritter:False):
    pm = PromptManager()
    intent_classifier = pm.render(
        "intent_classifier",
        query=query
    )
    query_refiner = pm.render(
        "query_refiner",
        query=query
    )
    res_dict={}
    intent_classifier_messages=[{"role": "system", "content": intent_classifier["system"]},
                {"role": "user", "content": intent_classifier["user"]}]
    intent_classifier_output = ask_ollama(
        prompts=intent_classifier_messages
    )
    res_dict["intent_classifier"]=intent_classifier_output
    if rewritter:
        query_refiner_messages=[{"role": "system", "content": query_refiner["system"]},
                    {"role": "user", "content": query_refiner["user"]}]
        query_refiner_output = ask_ollama(
            prompts=query_refiner_messages
        )
        res_dict["query_refiner"]=query_refiner_output
    return res_dict

if __name__ == "__main__":
    res_dict_list=[]
    queries = [
    # Simple single-file/single-function
    "请优化 utils.py 里的 load_config 函数，让它能处理 JSON5 格式。",
    "把 parser.py 中的 parse_line 函数的正则匹配速度优化一下，现在太慢了。",
    "修复 main.py 的 run 函数在空参数时会报错的问题。",
    "请解释 models.py 里的 User 类的 to_dict 方法作用是什么？",
    "我想给 db.py 增加一个函数，用于批量插入用户数据。",

    # Cross-file / cross-function
    "update_user 这个功能分散在多个文件里，请帮我查一下它最终调用链是怎样的？",
    "upload_service.py 的 upload_image 最终调用到了 storage/s3.py 的哪些函数？",
    "我改了 config.yaml 的路径，为什么 run.py 和 settings.py 都需要一起修改？帮我找一下相关依赖。",
    "login 的整个流程是怎样从 controller 到 service 再到 dao 的？我想在中间加一个审计日志。",

    # Refactor / structural change
    "现在 config_loader.py 和 config_utils.py 都处理配置文件，能帮我统一成一个模块吗？",
    "想把 message_queue 相关逻辑从 main.py 中拆出去，做成一个独立模块，帮我分析要改哪些地方。",
    "当前的 database.py 包含太多无关职责，能否帮我分拆成连接、迁移、操作三部分？",

    # Explanation + dependency
    "请解释 handler.py 的 process_request 是如何依赖 middlewares/ 目录下的内容的。",
    "api_router.py 注册了哪些路由？对应的 handler 分别在什么文件？",
    "帮我列出 models/ 下所有类之间的继承关系。",

    # Debug / behavior analysis
    "为什么调用 DataParser.parse 时会出现 NoneType 错误？帮我找出潜在原因。",
    "启动项目时报错：AttributeError: 'Config' object has no attribute 'reload'，请定位问题代码。",
    "导入 CSV 的时候，第三列如果为空会导致异常，请帮我分析发生在哪个函数。",

    # Fuzzy requirement
    "我想给用户系统加一个“冻结账号”的功能。",
    "能不能让日志更详细一点？",
    "我想让登录速度变快，你看看怎么优化？",
    "能否让这个接口支持更多格式？",

    # Code generation / NL2Code
    "请在 utils/string_utils.py 中新增一个 slugify 函数，把字符串转成 URL 安全格式。",
    "给 database/migration.py 写一个函数，用于自动执行未运行的 migration 文件。",

    # Complex tasks
    "请找出 process_data 在整个项目中的调用位置，然后检查每个调用点是否都处理了空值。如果没有，请生成具体修改建议。",
    "我想把 upload 和 download 逻辑合并成一个 FileService，请分析一下影响范围，并生成初步的设计草案。",
    "在用户注册时加入验证码校验，请帮我列出所有需要动的文件、函数，以及依赖链。",
]
    for query in queries:
        intent_dict=intent_detection(query)
        print(intent_dict)
        res_dict_list.append({"query":query, "intent_classifier":intent_dict["intent_classifier"],"query_refiner":intent_dict["query_refiner"]})
        
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(res_dict_list, f, ensure_ascii=False)



