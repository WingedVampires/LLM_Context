import json
import re
from openai import OpenAI
import os
from neo4j import GraphDatabase

api_key = 'XX'

uri = "bolt://127.0.0.1:7687"  # 确保使用 Bolt 协议和正确端口
user = "neo4j"
password = "88888888"

driver = GraphDatabase.driver(uri, auth=(user, password))

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

with (open('./prompts/1-planer.txt', 'r') as f_1,
      open('./prompts/2-analysis.txt', 'r') as f_2,
      open('./prompts/3-sufficient.txt', 'r') as f_3):
    planer_prompt = f_1.read()
    analysis_prompt = f_2.read()
    sufficient_prompt = f_3.read()

messages = []


def call_llm(message):
    messages.append({"role": "user", "content": message})
    resp = client.chat.completions.create(
        model="deepseek-v3.2-exp",  # 确认这个模型在 DashScope 兼容端可用
        messages=messages,
        extra_body={"enable_thinking": True},
        stream=False
    )

    # 一些兼容端会把“思考过程”放在额外字段：尝试读取（若没有就忽略）
    rc = getattr(resp.choices[0], "reasoning_content", None)
    if rc:
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        print(rc)

    return resp.choices[0].message.content


def extract_all(text):
    # Extract Thought
    thought_pattern = r"\*\*Thought:\*\*(.*?)(?=\*\*Cypher|\*\*Natural|\Z)"
    thought = re.findall(thought_pattern, text, re.DOTALL)
    thought = thought[0].strip() if thought else None

    # Extract Cypher Queries (multiple blocks)
    cypher_pattern = r"```cypher(.*?)```"
    cypher_queries = [m.strip() for m in re.findall(cypher_pattern, text, re.DOTALL)]

    # Extract Natural Language Queries (multiple lines until next section)
    nl_pattern = r"Natural Language Queries:\*\*(.*)"
    nl_matches = re.findall(nl_pattern, text, re.DOTALL)

    natural_queries = []
    if nl_matches:
        raw_block = nl_matches[0].strip()

        # Split into multiple queries by newline
        for line in raw_block.split("\n"):
            cleaned = line.strip()
            if cleaned:
                # Remove trailing Markdown leftovers
                cleaned = cleaned.rstrip("*").rstrip("_").strip()
                natural_queries.append(cleaned)

    return {
        "thought": thought,
        "cypher_queries": cypher_queries,
        "natural_language_queries": natural_queries
    }


def run_cypher(cypher):
    res = ''
    with driver.session() as session:
        records = session.run(cypher)
        # for rec in records:

        for rec in records:
            # for k, v in rec.items():
            #     print(k)
            #     print(v)
            return rec


def cypher_respond_format(cypher_respond, query_id):
    res = (f"### Result {query_id}\n" +
           f"- query_id: {query_id}\n")
    for k, v in cypher_respond.items():
        res += f"- {k}: {v}\n"
    return res


if __name__ == '__main__':
    new_res = '''**Thought:** The retrieval provides the code for the `randomCluster` method, but it shows inconsistent usage of `relationships` (e.g., passed directly to `randomInstance` in some calls, while others use `Arrays.asList` of specific relationships). To determine if `relationships` is a typo for `relationship`, I need to examine the `GraphManager` class definition for field declarations and the `randomInstance` method signature to understand expected parameters.

    **Answers:**
    1. Yes, I need more context to verify if `relationships` is a valid field or a typo.
    2. No, the current information is not enough to answer the raw question definitively.

    **New Cypher Queries:**
    ```cypher
    MATCH (c:Class {name: 'org.example.gdsmith.cypher.gen.GraphManager'})
    RETURN c.name AS name, c.source_code AS code, c.signature AS signature;
    ```

    ```cypher
    MATCH (c:Class {name: 'org.example.gdsmith.cypher.gen.GraphManager'})-[:RELATED {description: "contains method"}]->(m:Method {name: 'randomInstance'})
    RETURN m.name AS name, m.signature AS signature, m.source_code AS code;
    ```

    **New Natural Language Queries:**
    The class named 'org.example.gdsmith.cypher.gen.GraphManager' and its source code.
    The method named 'randomInstance' contained in the class 'org.example.gdsmith.cypher.gen.GraphManager'.

    '''
    print(json.dumps(extract_all(new_res), indent=4))
    exit(0)
    # response = call_llm(
    #     {"role": "user", "content": planer_prompt.format(
    #        user_query= 'Is this really a relationships, not a relationship? It may be a code error\n\nIn the randomCluster method of the GraphManager file, this code example. add (randomInstance (Arrays. asList (relationships. getFrom(), relationships. getTo()), relationships); This should be a relationship, right？the GraphManager file is src/main/java/org/example/gdsmith/cypher/gen/GraphManager.java')
    #      }
    # )

    res = '''**Thought:** No incomplete element identified. The issue is about a potential typo in variable usage within an existing method. We need to locate the specific method and file to examine the code.

**Cypher Queries:**
```cypher
MATCH (m:Method {name: 'randomCluster'})
RETURN m.name AS name, m.source_code AS code, m.signature AS signature;
```

**Natural Language Queries:**
The method named 'randomCluster' in file 'src/main/java/org/example/gdsmith/cypher/gen/GraphManager.java'''
    re = extract_all(res)
    cypher_respond_txt = ''
    for query_idx, cypher in enumerate(re["cypher_queries"]):
        cypher_respond = run_cypher(cypher)
        cypher_respond_txt += cypher_respond_format(cypher_respond, query_id=query_idx)

    messages.append({"role": "user", "content": planer_prompt.format(
        user_query='Is this really a relationships, not a relationship? It may be a code error\n\nIn the randomCluster method of the GraphManager file, this code example. add (randomInstance (Arrays. asList (relationships. getFrom(), relationships. getTo()), relationships); This should be a relationship, right？the GraphManager file is src/main/java/org/example/gdsmith/cypher/gen/GraphManager.java')
                     })
    messages.append({"role": "assistant", "content": res})
    # response = call_llm(analysis_promptlysis_prompt.format(cypher_search_result=cypher_respond_txt, natural_language_search_result='None'))
    new_res ='''**Thought:** The issue involves a potential typo in the randomCluster method where "relationships" is used but not defined. It should likely be replaced with a list containing the single "relationship" (e.g., `Arrays.asList(relationship)`). Additionally, understanding the randomInstance method's signature could confirm the expected parameters for the fix.

**Cypher Queries:**
```cypher
MATCH (m:Method {name: 'randomInstance'})
RETURN m.name AS name, m.source_code AS code, m.signature AS signature;
```

**Natural Language Queries:**
Find the method named 'randomInstance' to verify its parameter expectations and confirm the correct variable usage.
'''
    new_res = '''**Thought:** The retrieval provides the code for the `randomCluster` method, but it shows inconsistent usage of `relationships` (e.g., passed directly to `randomInstance` in some calls, while others use `Arrays.asList` of specific relationships). To determine if `relationships` is a typo for `relationship`, I need to examine the `GraphManager` class definition for field declarations and the `randomInstance` method signature to understand expected parameters.

**Answers:**
1. Yes, I need more context to verify if `relationships` is a valid field or a typo.
2. No, the current information is not enough to answer the raw question definitively.

**New Cypher Queries:**
```cypher
MATCH (c:Class {name: 'org.example.gdsmith.cypher.gen.GraphManager'})
RETURN c.name AS name, c.source_code AS code, c.signature AS signature;
```

```cypher
MATCH (c:Class {name: 'org.example.gdsmith.cypher.gen.GraphManager'})-[:RELATED {description: "contains method"}]->(m:Method {name: 'randomInstance'})
RETURN m.name AS name, m.signature AS signature, m.source_code AS code;
```

**New Natural Language Queries:**
The class named 'org.example.gdsmith.cypher.gen.GraphManager' and its source code.
The method named 'randomInstance' contained in the class 'org.example.gdsmith.cypher.gen.GraphManager'.

'''
    print(json.dumps(extract_all(new_res), indent=4))
    # print(messages)
    # print(response)
