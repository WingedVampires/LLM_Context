Please run the following command to construct the knowledge graph of the project `repos/GDsmith` in the Neo4j database:

```bash
cd ./Context_KG
python pipeline/construct_graph.py
```
The code `GDsmith` can be downloaded from https://github.com/ddaa2000/GDsmith. 
After you have downloaded the target repository, you should save it into the directory `Context_KG/repos/`.


uvicorn app.main:app --host 0.0.0.0 --port 6329