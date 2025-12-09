import json
import re
import time
from typing import List, Dict

from numpy.matlib import empty
from openai import OpenAI
import os
from neo4j import GraphDatabase

from embedding import Embedding
from tree_propagate import graph_retrieval

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
      open('./prompts/2-analysis.txt', 'r') as f_2):
    planer_prompt = f_1.read()
    analysis_prompt = f_2.read()

messages = []


def call_llm(message, is_thinking=False, MAX_HISTORY=5):
    # messages.append({"role": "user", "content": message})
    messages.append(message)

    tmp_messages = messages[-MAX_HISTORY:]

    resp = client.chat.completions.create(
        model="deepseek-v3.2-exp",  # 确认这个模型在 DashScope 兼容端可用
        messages=tmp_messages,
        extra_body={"enable_thinking": is_thinking},
        stream=False
    )

    # 一些兼容端会把“思考过程”放在额外字段：尝试读取（若没有就忽略）
    rc = getattr(resp.choices[0], "reasoning_content", None)
    if rc:
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
        print(rc)
    response = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    print(response)
    return response


def extract_query_results(text):
    # 1. Thought
    thought_pattern = r"\*\*Thought:\*\*(.*?)(?=\*\*Cypher|\*\*Natural|\Z)"
    thought_match = re.findall(thought_pattern, text, re.DOTALL | re.IGNORECASE)
    thought = thought_match[0].strip() if thought_match else None

    # 2. Cypher Queries
    cypher_pattern = r"([`']{3})\s*cypher(.*?)(?:\1)"
    cypher_blocks = re.findall(cypher_pattern, text, re.DOTALL | re.IGNORECASE)

    cypher_queries = []
    for fence, content in cypher_blocks:
        block = content.strip()
        if not block:
            continue

        # 在分号后跟 MATCH 或结尾处拆分
        parts = re.split(r";\s*(?=MATCH\b|$)", block, flags=re.IGNORECASE)

        for part in parts:
            q = part.strip()
            if not q:
                continue
            if not q.endswith(";"):
                q += ";"
            cypher_queries.append(q)

    # 3. Natural Language Queries
    nl_pattern = r"([`']{3})\s*natural language(.*?)(?:\1)"
    nl_blocks = re.findall(nl_pattern, text, re.DOTALL | re.IGNORECASE)

    natural_queries = []
    for fence, content in nl_blocks:
        block = content.strip()
        if not block:
            continue

        # 按空行拆成多条独立 NL 查询
        parts = re.split(r"\n\s*\n", block)
        for part in parts:
            q = part.strip()
            if q:
                natural_queries.append(q)

    return {
        "thought": thought,
        "cypher_queries": cypher_queries,
        "natural_language_queries": natural_queries
    }


def extract_analysis_results(text):
    pattern = r"##\s*(Q\d+)\s+answer:\s*(\w+)"
    matches = re.findall(pattern, text)

    results = {}
    for q, ans in matches:
        results[q] = True if ans.lower() == "yes" else False

    return results


def run_cypher(cypher):
    res = []
    with driver.session() as session:
        records = session.run(cypher)

        for rec in records:
            res.append(rec)

    return res


def cypher_respond_format(cypher_respond, query_id):
    res = (f"### Result {query_id}\n" +
           f"- query_id: {query_id}\n")
    for k, v in cypher_respond.items():
        res += f"- {k}: {v}\n"
    return res


def retrieve(raw_query, semantic_search_top_k=5, save_path='res') -> Dict[str, Dict]:
    retrieval_results: Dict = {'cypher': {}, 'nl': {}}
    import os
    os.makedirs(save_path, exist_ok=True)
    response = call_llm(
        {"role": "user", "content": planer_prompt.format(
            user_query=raw_query)
         }
    )
    with open(f'{save_path}/planner_input.txt', 'w+') as f:
        f.write(planer_prompt.format(
            user_query=raw_query)
        )
    with open(f'{save_path}/planner_output.txt', 'w+') as f:
        f.write(response)

    # with open(f'{path}/planner.txt', 'r') as f:
    #     response = f.read()

    epoch = 0

    while epoch < 10:
        cypher_respond_txt = ''
        natural_language_respond_txt = ''

        query_results = extract_query_results(response)

        for query_idx, (cypher, natural_language) in enumerate(zip(query_results["cypher_queries"], query_results["natural_language_queries"])):
            #### 根据llm提出的query进行cypher查询
            cypher_responds = run_cypher(cypher)
            candidate_nodes_ids = []
            st_time = time.time()
            #### 根据llm提出的query进行候选图语义查询
            for cypher_respond in cypher_responds:
                node_id = cypher_respond["id"]
                candidate_nodes_ids.append(node_id)
                cypher_respond_txt += cypher_respond_format(cypher_respond, query_id=query_idx)

                retrieval_results['cypher'][node_id] = cypher_respond
            sc_time = time.time()
            duration = sc_time - st_time
            print(f"cypher retrieval duration: {duration}")
            print(len(candidate_nodes_ids))
            if len(candidate_nodes_ids) != 0:
                query_embedding: List[float] = embedding_tool.get_embedding(natural_language)
                top_nodes, top_scores, top_node_names, top_node_codes, top_k_file_paths = graph_retrieval(
                    driver=driver,
                    candidate_ids=candidate_nodes_ids,
                    query_embedding=query_embedding,
                    k=semantic_search_top_k,
                    max_hop=2,
                    alpha=0.6,
                    beta=0.2,
                    gamma=0.2,
                    T=2,
                )
                duration = time.time() - sc_time
                print(f"tree search retrieval duration: {duration}")
                for i, node_id in enumerate(top_nodes):
                    if node_id not in candidate_nodes_ids:
                        natural_language_respond_txt += cypher_respond_format({'name': top_node_names[i], 'code': top_node_codes[node_id], 'file_path': top_k_file_paths[node_id]}, query_id=query_idx)

                        retrieval_results['nl'][node_id] = {'name': top_node_names[i], 'code': top_node_codes[node_id], 'file_path': top_k_file_paths[node_id]}

        with open(f'{save_path}/analysis-{epoch}_input.txt', 'w+') as f:
            f.write(analysis_prompt.format(cypher_search_result=cypher_respond_txt,
                                           user_query=raw_query,
                                           natural_language_search_result=natural_language_respond_txt))

        response = call_llm({"role": "user", "content": analysis_prompt.format(cypher_search_result=cypher_respond_txt,
                                                                               user_query=raw_query,
                                                                               natural_language_search_result=natural_language_respond_txt)})
        with open(f'{save_path}/analysis-{epoch}_output.txt', 'w+') as f:
            f.write(response)

        analysis_result = extract_analysis_results(response)

        if analysis_result['Q2'] or not analysis_result['Q1']:
            return retrieval_results

        epoch += 1

    return retrieval_results


if __name__ == '__main__':
    upstream_input = {
        "hash": "7ec7dd1a19a38a1829903f88b9512697bfb7be3c",
        "author": "Les Hazlewood <121180+lhazlewood@users.noreply.github.com>",
        "date": "Wed Aug 13 15:18:03 2025 -0400",
        "message": "Enable JwtParser empty nested algorithm collections. (#1007)\n\nResolves #996.\n\nAllowed the JwtParser to have empty nested algorithm collections, effectively disabling the parser's associated feature:\n- Clearing the zip() nested collection means parser decompression is disabled\n- Clearing the sig() nested collection means parser signature verification is disabled (i.e. all JWSs will be unsupported/rejected)\n- Clearing the enc() or key() nested collections means parser decryption is disabled (i.e. all JWEs will be unsupported/rejected)",
        "merge": "",
        "files": [
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/DefaultHeader.java",
                "added": 7,
                "deleted": 0,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/DefaultHeader.java b/impl/src/main/java/io/jsonwebtoken/impl/DefaultHeader.java\nindex 94cc799..453a91e 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/DefaultHeader.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/DefaultHeader.java\n@@ -17,8 +17,10 @@ package io.jsonwebtoken.impl;\n \n import io.jsonwebtoken.Header;\n import io.jsonwebtoken.impl.lang.CompactMediaTypeIdConverter;\n+import io.jsonwebtoken.impl.lang.Nameable;\n import io.jsonwebtoken.impl.lang.Parameter;\n import io.jsonwebtoken.impl.lang.Parameters;\n+import io.jsonwebtoken.lang.Assert;\n import io.jsonwebtoken.lang.Registry;\n import io.jsonwebtoken.lang.Strings;\n \n@@ -54,6 +56,11 @@ public class DefaultHeader extends ParameterMap implements Header {\n         return \"JWT header\";\n     }\n \n+    static String nameOf(Header header) {\n+        return Assert.hasText(Assert.isInstanceOf(Nameable.class, header).getName(),\n+                \"Header name cannot be null or empty.\");\n+    }\n+\n     @Override\n     public String getType() {\n         return get(TYPE);\n"
            },
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/DefaultJwtParser.java",
                "added": 13,
                "deleted": 16,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/DefaultJwtParser.java b/impl/src/main/java/io/jsonwebtoken/impl/DefaultJwtParser.java\nindex 75faa41..622f5d7 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/DefaultJwtParser.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/DefaultJwtParser.java\n@@ -241,11 +241,11 @@ public class DefaultJwtParser extends AbstractParser<Jwt<?, ?>> implements JwtPa\n         this.expectedClaims = Jwts.claims().add(expectedClaims);\n         this.decoder = Assert.notNull(base64UrlDecoder, \"base64UrlDecoder cannot be null.\");\n         this.deserializer = Assert.notNull(deserializer, \"JSON Deserializer cannot be null.\");\n-        this.sigAlgs = new IdLocator<>(DefaultHeader.ALGORITHM, sigAlgs, MISSING_JWS_ALG_MSG);\n-        this.keyAlgs = new IdLocator<>(DefaultHeader.ALGORITHM, keyAlgs, MISSING_JWE_ALG_MSG);\n-        this.encAlgs = new IdLocator<>(DefaultJweHeader.ENCRYPTION_ALGORITHM, encAlgs, MISSING_ENC_MSG);\n+        this.sigAlgs = new IdLocator<>(DefaultHeader.ALGORITHM, sigAlgs, \"mac or signature\", \"signature verification\", MISSING_JWS_ALG_MSG);\n+        this.keyAlgs = new IdLocator<>(DefaultHeader.ALGORITHM, keyAlgs, \"key management\", \"decryption\", MISSING_JWE_ALG_MSG);\n+        this.encAlgs = new IdLocator<>(DefaultJweHeader.ENCRYPTION_ALGORITHM, encAlgs, \"encryption\", \"decryption\", MISSING_ENC_MSG);\n         this.zipAlgs = compressionCodecResolver != null ? new CompressionCodecLocator(compressionCodecResolver) :\n-                new IdLocator<>(DefaultHeader.COMPRESSION_ALGORITHM, zipAlgs, null);\n+                new IdLocator<>(DefaultHeader.COMPRESSION_ALGORITHM, zipAlgs, \"compression\", \"decompression\", null);\n     }\n \n     @Override\n@@ -275,7 +275,7 @@ public class DefaultJwtParser extends AbstractParser<Jwt<?, ?>> implements JwtPa\n             algorithm = (SecureDigestAlgorithm<?, Key>) sigAlgs.apply(jwsHeader);\n         } catch (UnsupportedJwtException e) {\n             //For backwards compatibility.  TODO: remove this try/catch block for 1.0 and let UnsupportedJwtException propagate\n-            String msg = \"Unsupported signature algorithm '\" + alg + \"'\";\n+            String msg = \"Unsupported signature algorithm '\" + alg + \"': \" + e.getMessage();\n             throw new SignatureException(msg, e);\n         }\n         Assert.stateNotNull(algorithm, \"JWS Signature Algorithm cannot be null.\");\n@@ -459,7 +459,7 @@ public class DefaultJwtParser extends AbstractParser<Jwt<?, ?>> implements JwtPa\n         final boolean payloadBase64UrlEncoded = !(header instanceof JwsHeader) || ((JwsHeader) header).isPayloadEncoded();\n         if (payloadBase64UrlEncoded) {\n             // standard encoding, so decode it:\n-            byte[] data = decode(tokenized.getPayload(), \"payload\");\n+            byte[] data = decode(payloadToken, \"payload\");\n             payload = new Payload(data, header.getContentType());\n         } else {\n             // The JWT uses the b64 extension, and we already know the parser supports that extension at this point\n@@ -493,6 +493,13 @@ public class DefaultJwtParser extends AbstractParser<Jwt<?, ?>> implements JwtPa\n             TokenizedJwe tokenizedJwe = (TokenizedJwe) tokenized;\n             JweHeader jweHeader = Assert.stateIsInstance(JweHeader.class, header, \"Not a JweHeader. \");\n \n+            // Ensure both an 'alg' and 'enc' header value exists and is supported before spending time/effort\n+            // base64Url-decoding anything:\n+            final AeadAlgorithm encAlg = this.encAlgs.apply(jweHeader);\n+            Assert.stateNotNull(encAlg, \"JWE Encryption Algorithm cannot be null.\");\n+            @SuppressWarnings(\"rawtypes\") final KeyAlgorithm keyAlg = this.keyAlgs.apply(jweHeader);\n+            Assert.stateNotNull(keyAlg, \"JWE Key Algorithm cannot be null.\");\n+\n             byte[] cekBytes = Bytes.EMPTY; //ignored unless using an encrypted key algorithm\n             CharSequence base64Url = tokenizedJwe.getEncryptedKey();\n             if (Strings.hasText(base64Url)) {\n@@ -529,16 +536,6 @@ public class DefaultJwtParser extends AbstractParser<Jwt<?, ?>> implements JwtPa\n                 throw new MalformedJwtException(msg);\n             }\n \n-            String enc = jweHeader.getEncryptionAlgorithm();\n-            if (!Strings.hasText(enc)) {\n-                throw new MalformedJwtException(MISSING_ENC_MSG);\n-            }\n-            final AeadAlgorithm encAlg = this.encAlgs.apply(jweHeader);\n-            Assert.stateNotNull(encAlg, \"JWE Encryption Algorithm cannot be null.\");\n-\n-            @SuppressWarnings(\"rawtypes\") final KeyAlgorithm keyAlg = this.keyAlgs.apply(jweHeader);\n-            Assert.stateNotNull(keyAlg, \"JWE Key Algorithm cannot be null.\");\n-\n             Key key = this.keyLocator.locate(jweHeader);\n             if (key == null) {\n                 String msg = \"Cannot decrypt JWE payload: unable to locate key for JWE with header: \" + jweHeader;\n"
            },
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/IdLocator.java",
                "added": 22,
                "deleted": 22,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/IdLocator.java b/impl/src/main/java/io/jsonwebtoken/impl/IdLocator.java\nindex 3216171..88b8df2 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/IdLocator.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/IdLocator.java\n@@ -17,8 +17,6 @@ package io.jsonwebtoken.impl;\n \n import io.jsonwebtoken.Header;\n import io.jsonwebtoken.Identifiable;\n-import io.jsonwebtoken.JweHeader;\n-import io.jsonwebtoken.JwsHeader;\n import io.jsonwebtoken.Locator;\n import io.jsonwebtoken.MalformedJwtException;\n import io.jsonwebtoken.UnsupportedJwtException;\n@@ -31,38 +29,27 @@ import io.jsonwebtoken.lang.Strings;\n public class IdLocator<H extends Header, R extends Identifiable> implements Locator<R>, Function<H, R> {\n \n     private final Parameter<String> param;\n-    private final String requiredMsg;\n-    private final boolean valueRequired;\n-\n     private final Registry<String, R> registry;\n+    private final String algType;\n+    private final String behavior;\n+    private final String requiredMsg;\n \n-    public IdLocator(Parameter<String> param, Registry<String, R> registry, String requiredExceptionMessage) {\n+    public IdLocator(Parameter<String> param, Registry<String, R> registry, String algType, String behavior, String requiredExceptionMessage) {\n         this.param = Assert.notNull(param, \"Header param cannot be null.\");\n+        this.registry = Assert.notNull(registry, \"Registry cannot be null.\");\n+        this.algType = Assert.hasText(algType, \"algType cannot be null or empty.\");\n+        this.behavior = Assert.hasText(behavior, \"behavior cannot be null or empty.\");\n         this.requiredMsg = Strings.clean(requiredExceptionMessage);\n-        this.valueRequired = Strings.hasText(this.requiredMsg);\n-        Assert.notEmpty(registry, \"Registry cannot be null or empty.\");\n-        this.registry = registry;\n-    }\n-\n-    private static String type(Header header) {\n-        if (header instanceof JweHeader) {\n-            return \"JWE\";\n-        } else if (header instanceof JwsHeader) {\n-            return \"JWS\";\n-        } else {\n-            return \"JWT\";\n-        }\n     }\n \n     @Override\n     public R locate(Header header) {\n-        Assert.notNull(header, \"Header argument cannot be null.\");\n \n         Object val = header.get(this.param.getId());\n         String id = val != null ? val.toString() : null;\n \n         if (!Strings.hasText(id)) {\n-            if (this.valueRequired) {\n+            if (this.requiredMsg != null) { // a msg was provided, so the value is required:\n                 throw new MalformedJwtException(requiredMsg);\n             }\n             return null; // otherwise header value not required, so short circuit\n@@ -71,7 +58,20 @@ public class IdLocator<H extends Header, R extends Identifiable> implements Loca\n         try {\n             return registry.forKey(id);\n         } catch (Exception e) {\n-            String msg = \"Unrecognized \" + type(header) + \" \" + this.param + \" header value: \" + id;\n+            StringBuilder sb = new StringBuilder(\"Unsupported \")\n+                    .append(DefaultHeader.nameOf(header))\n+                    .append(\" \")\n+                    .append(this.param)\n+                    .append(\" value '\").append(id).append(\"'\");\n+            if (this.registry.isEmpty()) {\n+                sb.append(\": \")\n+                        .append(this.behavior)\n+                        .append(\" is disabled (no \")\n+                        .append(this.algType)\n+                        .append(\" algorithms have been configured)\");\n+            }\n+            sb.append(\".\");\n+            String msg = sb.toString();\n             throw new UnsupportedJwtException(msg, e);\n         }\n     }\n"
            },
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/lang/DefaultRegistry.java",
                "added": 2,
                "deleted": 2,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/lang/DefaultRegistry.java b/impl/src/main/java/io/jsonwebtoken/impl/lang/DefaultRegistry.java\nindex 98cee83..0b48873 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/lang/DefaultRegistry.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/lang/DefaultRegistry.java\n@@ -29,9 +29,9 @@ public class DefaultRegistry<K, V> extends DelegatingMap<K, V, Map<K, V>> implem\n     private final String qualifiedKeyName;\n \n     private static <K, V> Map<K, V> toMap(Collection<? extends V> values, Function<V, K> keyFn) {\n-        Assert.notEmpty(values, \"Collection of values may not be null or empty.\");\n+        Assert.notNull(values, \"Collection of values may not be null.\");\n         Assert.notNull(keyFn, \"Key function cannot be null.\");\n-        Map<K, V> m = new LinkedHashMap<>(values.size());\n+        Map<K, V> m = new LinkedHashMap<>(Collections.size(values));\n         for (V value : values) {\n             K key = Assert.notNull(keyFn.apply(value), \"Key function cannot return a null value.\");\n             m.put(key, value);\n"
            },
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/lang/IdRegistry.java",
                "added": 1,
                "deleted": 1,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/lang/IdRegistry.java b/impl/src/main/java/io/jsonwebtoken/impl/lang/IdRegistry.java\nindex d49c570..91f062f 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/lang/IdRegistry.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/lang/IdRegistry.java\n@@ -53,7 +53,7 @@ public class IdRegistry<T extends Identifiable> extends StringRegistry<T> {\n \n     public IdRegistry(String name, Collection<T> instances, boolean caseSensitive) {\n         super(name, \"id\",\n-                Assert.notEmpty(instances, \"Collection of Identifiable instances may not be null or empty.\"),\n+                Assert.notNull(instances, \"Collection of Identifiable instances may not be null.\"),\n                 IdRegistry.<T>fn(),\n                 caseSensitive);\n     }\n"
            },
            {
                "file": "impl/src/main/java/io/jsonwebtoken/impl/lang/StringRegistry.java",
                "added": 1,
                "deleted": 1,
                "diff": "diff --git a/impl/src/main/java/io/jsonwebtoken/impl/lang/StringRegistry.java b/impl/src/main/java/io/jsonwebtoken/impl/lang/StringRegistry.java\nindex efdb917..ea22672 100644\n--- a/impl/src/main/java/io/jsonwebtoken/impl/lang/StringRegistry.java\n+++ b/impl/src/main/java/io/jsonwebtoken/impl/lang/StringRegistry.java\n@@ -37,7 +37,7 @@ public class StringRegistry<V> extends DefaultRegistry<String, V> {\n     public StringRegistry(String name, String keyName, Collection<V> values, Function<V, String> keyFn, Function<String, String> caseFn) {\n         super(name, keyName, values, keyFn);\n         this.CASE_FN = Assert.notNull(caseFn, \"Case function cannot be null.\");\n-        Map<String, V> m = new LinkedHashMap<>(values().size());\n+        Map<String, V> m = new LinkedHashMap<>(Collections.size(values));\n         for (V value : values) {\n             String key = keyFn.apply(value);\n             key = this.CASE_FN.apply(key);\n"
            }
        ],
        "added": 175,
        "deleted": 60,
        "requirements": [
            "Allow the JwtParser to be configured with empty nested algorithm collections (zip, sig, enc, key) so that compression, signature verification, and encryption/decryption features can be disabled without causing errors.",
            "Provide clear, informative error messages when an unsupported algorithm is encountered or required algorithm headers are missing, explicitly stating the algorithm type and the disabled behavior.",
            "Validate the presence and support of required algorithms (e.g., JWE 'alg' and 'enc') before performing any base64 decoding or further processing, to avoid unnecessary work and improve failure handling."
        ]
    }
    embedding_tool = Embedding()
    raw_query: str = upstream_input['requirements'][0]

    retrieve(raw_query=raw_query, save_path='res-2')
    # with open(f'/root/lishuochuan/LLM_Context/res-1/analysis-3.txt', 'r') as f:
    #     response = f.read()
    # query_results = extract_query_results(response)
    # print(json.dumps(query_results, indent=4))
    # cypher_respond_txt = ''
    # for query_idx, (cypher, natural_language) in enumerate(zip(query_results["cypher_queries"], query_results["natural_language_queries"])):
    #     #### 根据llm提出的query进行cypher查询
    #     cypher_responds = run_cypher(cypher)
    #     candidate_nodes_ids = []
    #
    #     #### 根据llm提出的query进行候选图语义查询
    #     for cypher_respond in cypher_responds:
    #         node_id = cypher_respond["id"]
    #         candidate_nodes_ids.append(node_id)
    #         cypher_respond_txt += cypher_respond_format(cypher_respond, query_id=query_idx)
    # print(cypher_respond_txt)
    # exit(0)
