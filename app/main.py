from typing import Optional

from .intent_detection import intent_detection,subquestion_split

import torch
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile

from Context_KG.pipline.construct_graph import build_graph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




app = FastAPI(
    title="RAG_APP",
    description="代码RAG"
)

@app.get("/")
def index():
    return {"message": "Hello World"}



# the model initialized when the app gets loaded but we can configure it if we want
@app.get("/init_llm")
def init_llama_llm(n_gpu_layers: int = Query(500, description="Number of layers to load in GPU"),
                n_batch: int = Query(32, description="Number of tokens to process in parallel. Should be a number between 1 and n_ctx."),
                max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
                n_ctx: int = Query(4096, description="Token context window."),
                temperature: int = Query(0, description="Temperature for sampling. Higher values means more random samples.")):
    return

import os
import zipfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

UPLOAD_BASE_DIR = "./repos"

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: Optional[str] = "test_collection"
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip files are allowed."
        )

    collection_dir = os.path.join(UPLOAD_BASE_DIR, collection_name)
    os.makedirs(collection_dir, exist_ok=True)

    zip_path = os.path.join(collection_dir, file.filename)
    try:
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    finally:
        await file.close()

    extract_to = os.path.join(collection_dir)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                member_path = os.path.abspath(os.path.join(extract_to, member))
                if not member_path.startswith(os.path.abspath(extract_to)):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid zip file: contains path traversal attempt."
                    )
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid ZIP archive."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract ZIP file: {str(e)}"
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "File uploaded and extracted successfully.",
            "collection_name": collection_name,
            "extracted_path": extract_to
        }
    )

@app.post("/build_graph")
def graph_build(repo_path = '/home/hreyulog/codebase/LLM_Context/Context_KG/repos/GDsmith', language = 'java', output_dir = '../output'):
    try:
        build_graph(repo_path = repo_path, language = language, output_dir = output_dir)
    except Exception:
        return {"message": "Failed build graph"}
    return {"message": f"Successfully build graph"}

@app.post("/upload_and_build_graph")
def upload_and_build_graph(file: UploadFile = File(...),
    collection_name: Optional[str] = "test_collection",
    language = 'java', 
    output_dir = '../output'
    ):
    try:
        upload_file(file, collection_name)
        print('success upload')
        build_graph(repo_path = f'./repos/{collection_name}', language = language, output_dir = output_dir)
    except Exception:
        return {"message": "Failed build graph"}
    return {"message": f"Successfully build graph"}
    
@app.get("/query")
def query(query : str):
    subquery=eval(subquestion_split(query))
    list_intent=[]
    print(subquery)
    if subquery['need_decomposition']=='yes':
        for subq in subquery['subqueries']:
            dict_intent = intent_detection(subq,False)
            list_intent.append(dict_intent)
    else:
        dict_intent = intent_detection(query,False)
        list_intent.append(dict_intent)
    return list_intent


if __name__ == "__main__":
    pass

