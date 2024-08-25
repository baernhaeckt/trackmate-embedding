import torch

from PIL import Image
from services.embedding_service import EmbeddingService

if __name__ == '__main__':
    embedding_service = EmbeddingService()

    input_image_1 = Image.open("WhatsApp Image 2024-08-24 at 22.10.04.jpeg")
    input_image_2 = Image.open("WhatsApp Image 2024-08-24 at 22.10.07.jpeg")

    embedding_1 = embedding_service.create_embedding(input_image_1).unsqueeze(0)
    embedding_2 = embedding_service.create_embedding(input_image_2).unsqueeze(0)

    sim = torch.nn.functional.cosine_similarity(embedding_1, embedding_2)
    print(sim)