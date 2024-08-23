import shutil
import torch

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/create", tags=["embedding"], status_code=200)
def create_embedding(file: UploadFile):
    suffix = Path(file.filename).suffix

    # Load the Model from pytorch hub
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
    embedding_model.eval()

    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)

        input_image = Image.open(tmp.name)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            model.to("cuda")

        with torch.no_grad():
            output = embedding_model(input_batch).squeeze()

    return JSONResponse(content={"embedding": output.detach().cpu().numpy().tolist()})
