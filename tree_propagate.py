from typing import *

from neo4j import GraphDatabase
from collections import defaultdict, deque
import math

from embedding import Embedding

NodeId = Hashable

EDGE_WEIGHT_SCORES = {}
LABEL_RANK = {
    "Project": 5,
    "Subproject": 4,
    "Feature": 3,
    "Chain": 2,
    "Class": 1,
    "Method": 0,
}


def propagate_scores(
        nodes,  # list of node ids
        init_scores,  # dict: node_id → s_init
        in_neighbors,  # dict: node_id → list of (neighbor_id, weight_u_to_v)
        out_neighbors,  # dict: node_id → list of (neighbor_id, weight_v_to_u)
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
        T=2
):
    """
    nodes: 所有节点的列表
    init_scores: 每个节点的初始得分 s_init,v
    in_neighbors: 每个节点的上游邻居 (u->v) 列表，形式 [(u, w_u_to_v), ...]
    out_neighbors: 每个节点的下游邻居 (v->u) 列表，形式 [(u, w_v_to_u), ...]
    alpha, beta, gamma: 参数（alpha + beta + gamma = 1）
    T: 传播轮数
    """
    # 初始化 S_v^{(0)}
    S = {v: init_scores.get(v, 0.0) for v in nodes}

    for t in range(T):
        S_new = {}
        for v in nodes:
            s_init_v = init_scores.get(v, 0.0)

            # 上游贡献
            in_list = in_neighbors.get(v, [])
            denom_in = max(1, len(in_list))
            sum_in = sum(EDGE_WEIGHT_SCORES.get(w, 1) * S[u] for (u, w) in in_list)
            contrib_in = (sum_in / denom_in) if denom_in > 0 else 0.0

            # 下游贡献
            out_list = out_neighbors.get(v, [])
            denom_out = max(1, len(out_list))
            sum_out = sum(EDGE_WEIGHT_SCORES.get(w, 1) * S[u] for (u, w) in out_list)
            contrib_out = (sum_out / denom_out) if denom_out > 0 else 0.0

            # 新得分
            S_new[v] = alpha * s_init_v + beta * contrib_in + gamma * contrib_out

        # 更新
        S = S_new

    return S


def build_candidate_graph_from_neo4j(
        driver,
        candidate_ids: List[int],
        max_hop: int = 2,
        label_rank: Dict[str, int] = LABEL_RANK,
) -> Tuple[List[NodeId], List[Tuple[NodeId, NodeId, float]], Dict[NodeId, List[str]], Dict[NodeId, List[float]], Dict[NodeId, str], Dict[NodeId, str], Dict[NodeId, str]]:
    """
    从一批候选节点出发，做 1..max_hop 跳扩展，得到：
    - nodes: 所有涉及的节点 id 列表
    - edges: (u, v, weight) 列表，其中方向满足 “高层级 → 低层级”，
             且同一对节点只保留一条边（去掉 A->B 与 B->A 的重复）
    - node_labels: 节点 id → labels 列表
    """
    with driver.session() as session:
        # 1) 收集节点及其 labels
        node_records = session.run(
            f"""
            MATCH (c)
            WHERE id(c) IN $candidate_ids
            MATCH p = (c)-[*1..{max_hop}]-(n)
            WITH COLLECT(DISTINCT c) + COLLECT(DISTINCT n) AS nodes
            UNWIND nodes AS n
            RETURN DISTINCT id(n) AS id, n.name AS name, n.source_code AS code, n.file_path AS file_path, labels(n) AS labels, n.embedding AS embedding;
            """,
            candidate_ids=candidate_ids,
        )

        node_labels: Dict[NodeId, List[str]] = {}
        node_embeddings: Dict[NodeId, list] = {}
        node_names: Dict[NodeId, str] = {}
        node_codes: Dict[NodeId, str] = {}
        node_file_paths: Dict[NodeId, str] = {}

        for rec in node_records:
            node_id = rec["id"]
            labels = rec["labels"]
            embedding = rec["embedding"]
            name = rec["name"]
            code = rec["code"]
            file_path = rec["file_path"]
            node_labels[node_id] = labels
            node_embeddings[node_id] = embedding
            node_names[node_id] = name
            node_codes[node_id] = code
            node_file_paths[node_id] = file_path

        nodes = list(node_labels.keys())

        # 2) 收集所有原始有向边（可能有 A->B 和 B->A）
        edge_records = session.run(
            f"""
            MATCH (c)
            WHERE id(c) IN $candidate_ids
            MATCH p = (c)-[r*1..{max_hop}]-(n)
            UNWIND relationships(p) AS rel
            WITH DISTINCT rel AS r
            RETURN id(startNode(r)) AS s, id(endNode(r)) AS e, type(r) AS rel_type;
            """,
            candidate_ids=candidate_ids,
            max_hop=max_hop,
        )

        # 3) 基于层级重定向边方向，并对每一对节点去重
        #
        #   - 对每对节点 {A,B} 用无序 key = (min(A,B), max(A,B)) 去重，
        #     确保只保留一条边。
        #   - 方向：rank 高的作为 start，rank 低的作为 end；
        #           若 rank 相同，则保持第一次看到的方向。
        edge_map: Dict[Tuple[int, int], Tuple[int, int, float]] = {}

        for rec in edge_records:
            s = rec["s"]
            e = rec["e"]
            if s == e:
                # 自环一般没用，直接忽略
                continue

            # 这一对节点的无序 key
            key = (min(s, e), max(s, e))

            # 如果这一对已经处理过了，跳过（即去掉重复的反向边）
            if key in edge_map:
                continue

            # 计算两个节点的层级 rank
            rank_s = get_node_rank(s, node_labels, label_rank)
            rank_e = get_node_rank(e, node_labels, label_rank)

            # 按层级决定方向：高层级 -> 低层级
            if rank_s > rank_e:
                start, end = s, e
            elif rank_e > rank_s:
                start, end = e, s
            else:
                # 若层级相同：保持 Neo4j 原始方向 (s->e)
                start, end = s, e

            # 这里先简单设为 weight=1.0，你也可以按 rel_type 自定义
            edge_map[key] = (start, end, 1.0)

        edges = list(edge_map.values())

    return nodes, edges, node_labels, node_embeddings, node_names, node_codes, node_file_paths


def find_connected_components(nodes: List[NodeId], edges: List[Tuple[NodeId, NodeId, float]]):
    adj = defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components: List[List[NodeId]] = []

    for n in nodes:
        if n in visited:
            continue
        comp = []
        queue = deque([n])
        visited.add(n)
        while queue:
            x = queue.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    queue.append(y)
        components.append(comp)

    return components


def get_node_rank(node_id: NodeId, node_labels: Dict[NodeId, List[str]], label_rank: Dict[str, int]) -> int:
    labels = node_labels.get(node_id, [])
    if not labels:
        return -1  # 没 label 当成最低
    return max((label_rank.get(lbl, -1) for lbl in labels), default=-1)


def pick_top_node_for_component(
        comp_nodes: List[NodeId],
        node_labels: Dict[NodeId, List[str]],
        label_rank: Dict[str, int],
) -> NodeId:
    # 选“标签 rank 最大”的节点；若多个，取第一个
    best_node = None
    best_rank = -1
    for v in comp_nodes:
        rank = get_node_rank(node_id=v, node_labels=node_labels, label_rank=label_rank)
        if rank > best_rank:
            best_rank = rank
            best_node = v
    return best_node


def attach_virtual_root(
        nodes: List[NodeId],
        edges: List[Tuple[NodeId, NodeId, float]],
        node_labels: Dict[NodeId, List[str]],
        label_rank: Dict[str, int] = LABEL_RANK,
        root_id: NodeId = "VIRTUAL_ROOT",
) -> Tuple[List[NodeId], List[Tuple[NodeId, NodeId, float]], NodeId]:
    """
    对每个连通子图选一个最高等级节点，用虚拟根连接它：
        root -> top_node
    """
    components = find_connected_components(nodes, edges)

    all_nodes = list(nodes)
    all_edges = list(edges)

    # 加入虚拟根
    if root_id not in all_nodes:
        all_nodes.append(root_id)

    for comp in components:
        top_node = pick_top_node_for_component(comp, node_labels, label_rank)
        if top_node is not None:
            # 可以只连 root -> top_node，也可以连双向；传播时我们会对边双向建邻居
            all_edges.append((root_id, top_node, 1.0))

    return all_nodes, all_edges, root_id


def build_neighbors(
        nodes: List[NodeId],
        edges: List[Tuple[NodeId, NodeId, float]],
        bidirectional: bool = False,
):
    """
    根据边构建 in_neighbors / out_neighbors:
        in_neighbors[v]  = [(u, w_{u->v}), ...]
        out_neighbors[v] = [(u, w_{v->u}), ...]
    如果 bidirectional=True，则把每条边视为无向（双向传播）。
    """
    in_neighbors = {v: [] for v in nodes}
    out_neighbors = {v: [] for v in nodes}

    for u, v, w in edges:
        # u -> v
        out_neighbors[u].append((v, w))
        in_neighbors[v].append((u, w))

        if bidirectional:
            # v -> u
            out_neighbors[v].append((u, w))
            in_neighbors[u].append((v, w))

    return in_neighbors, out_neighbors


def compute_init_scores_from_embeddings(
        nodes: List[NodeId],
        query_emb,
        node_embeddings: Dict[NodeId, list],
) -> Dict[NodeId, float]:
    def cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    init_scores: Dict[NodeId, float] = {}
    for v in nodes:
        emb = node_embeddings.get(v)
        if emb is None:
            init_scores[v] = 0.0
        else:
            init_scores[v] = cosine(query_emb, emb)
    return init_scores


def graph_retrieval(
        driver,
        candidate_ids: List[int],
        query_embedding: list,
        k: int = 10,
        max_hop: int = 2,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        T: int = 2,
):
    """
    整体流程：
      1) 从候选节点出发，1..max_hop 跳扩展，构造候选图
      2) 构造虚拟根节点并连接每个候选子图的最高等级节点
      3) 用 query_embedding + 节点 embedding 计算 s_init
      4) 做 T 轮传播，得到最终 S_v
      5) 按 S_v 排序，返回 Top-K 节点及得分
    """
    # 1. 构建基础候选图
    nodes, edges, node_labels, node_embeddings, node_names, node_codes, node_file_paths = build_candidate_graph_from_neo4j(
        driver, candidate_ids, max_hop=max_hop
    )

    # 2. 附加虚拟根
    nodes_with_root, edges_with_root, root_id = attach_virtual_root(
        nodes, edges, node_labels, LABEL_RANK, root_id="VIRTUAL_ROOT"
    )

    # 3. 计算初始得分（含虚拟根：可以设为 0）
    init_scores = compute_init_scores_from_embeddings(
        nodes_with_root, query_embedding, node_embeddings
    )
    init_scores[root_id] = 0.0  # 虚拟根自己没有语义相关性

    # 4. 构建邻接结构
    in_neighbors, out_neighbors = build_neighbors(nodes_with_root, edges_with_root)

    # 5. 多轮传播
    final_scores = propagate_scores(
        nodes_with_root,
        init_scores,
        in_neighbors,
        out_neighbors,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        T=T,
    )

    # 6. 排序取 Top-K（通常不把虚拟根算在候选里）
    candidate_nodes_for_rank = [v for v in nodes_with_root if v != root_id]
    ranked = sorted(
        candidate_nodes_for_rank,
        key=lambda v: final_scores.get(v, 0.0),
        reverse=True,
    )

    top_k_nodes = ranked[:k]
    top_k_names = [node_names[node_id] for node_id in top_k_nodes]
    top_k_scores = {v: final_scores[v] for v in top_k_nodes}
    top_k_codes = {v: node_codes[v] for v in top_k_nodes}
    top_k_file_paths = {v: node_file_paths[v] for v in top_k_nodes}

    return top_k_nodes, top_k_scores, top_k_names, top_k_codes, top_k_file_paths


# 示例用法
if __name__ == "__main__":
    # TODO：注意要给图检索的node更高的权重

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "88888888"))
    embedding_tool = Embedding()

    # 1. 之前通过某种 Cypher 得到的初始候选节点 id 列表
    candidate_ids = [1566, 1577]

    # 2. query 的向量
    query_embedding: List[float] = embedding_tool.get_embedding(
        'Is this really a relationships, not a relationship? It may be a code error\n\nIn the randomCluster method of the GraphManager file, '
        'this code example. add (randomInstance (Arrays. asList (relationships. getFrom(), relationships. getTo()), relationships); '
        'This should be a relationship, right？the GraphManager file is src/main/java/org/example/gdsmith/cypher/gen/GraphManager.java')

    top_nodes, top_scores, top_node_names, top_node_codes = graph_retrieval(
        driver=driver,
        candidate_ids=candidate_ids,
        query_embedding=query_embedding,
        k=20,
        max_hop=2,
        alpha=0.6,
        beta=0.2,
        gamma=0.2,
        T=2,
    )

    print("Top-K 节点：", top_nodes)
    print("Top-K 节点名称：", top_node_names)
    print("对应得分：", top_scores)
