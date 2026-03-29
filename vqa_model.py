"""
VQA Model Architecture and Utilities
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple

class VQAModel(nn.Module):
    """Custom VQA model combining CNN and LSTM"""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_answers: int):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(512 + hidden_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_answers)

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        img_feat = self.cnn(image)
        q_embed = self.embedding(question)
        _, (h, _) = self.lstm(q_embed)
        q_feat = h.squeeze(0)
        combined = torch.cat((img_feat, q_feat), dim=1)
        x = self.relu(self.fc1(combined))
        out = self.fc2(x)
        return out
