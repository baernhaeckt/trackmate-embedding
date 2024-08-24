import torch

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


class EmbeddingService:
    def __init__(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.embedding_model.eval()

    def create_embedding(self, input_image):
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
            self.embedding_model.to("cuda")

        with torch.no_grad():
            output = self.embedding_model(input_batch).squeeze()

        return output
