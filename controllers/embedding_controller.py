import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image
from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

from services.embedding_service import EmbeddingService

router = APIRouter()


@router.post("/create", tags=["embedding"], status_code=200)
def create_embedding(file: UploadFile):
    suffix = Path(file.filename).suffix

    embedding_service = EmbeddingService()

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)

        input_image = Image.open(tmp.name)
        output = embedding_service.create_embedding(input_image)

    return JSONResponse(content={"embedding": output.detach().cpu().numpy().tolist()})
