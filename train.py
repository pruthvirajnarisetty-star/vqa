"""
Training script for custom VQA model
"""
import pandas as pd
import re
import pickle
from collections import Counter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vqa_model import VQAModel

class VQADataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row["image"].convert("RGB")
        image = self.transform(image)
        question = torch.tensor(row["question_encoded"], dtype=torch.long)
        answer = torch.tensor(row["answer_encoded"], dtype=torch.long)
        return image, question, answer

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text

def train_vqa_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = load_dataset("flaviagiammarino/vqa-rad")
    df = pd.DataFrame(dataset["train"])
    df = df[["image", "question", "answer"]]
    
    # Clean text
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)
    
    # Get top 50 answers
    top_answers = df["answer"].value_counts().nlargest(50).index
    df = df[df["answer"].isin(top_answers)]
    
    answer_to_idx = {a: i for i, a in enumerate(top_answers)}
    idx_to_answer = {i: a for a, i in answer_to_idx.items()}
    df["answer_encoded"] = df["answer"].apply(lambda x: answer_to_idx[x])
    
    # Build vocabulary
    vocab = {"<pad>": 0, "<unk>": 1}
    counter = Counter()
    for q in df["question"]:
        for w in q.split():
            counter[w] += 1
    
    idx = 2
    for word, count in counter.items():
        if count > 2:
            vocab[word] = idx
            idx += 1
    
    # Encode questions
    MAX_LEN = 20
    def encode_question(q):
        tokens = q.split()
        enc = [vocab.get(w, vocab["<unk>"]) for w in tokens]
        enc = enc[:MAX_LEN] + [vocab["<pad>"]] * (MAX_LEN - len(enc))
        return enc
    
    df["question_encoded"] = df["question"].apply(encode_question)
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    full_dataset = VQADataset(df, transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = VQAModel(len(vocab), 300, 256, len(answer_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, questions, answers in tqdm(train_loader):
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            
            outputs = model(images, questions)
            loss = criterion(outputs, answers)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "vqa_custom_model.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("answer_mapping.pkl", "wb") as f:
        pickle.dump(idx_to_answer, f)
    
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, questions, answers in test_loader:
            images = images.to(device)
            questions = questions.to(device)
            outputs = model(images, questions)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(answers.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1-Score  : {f1*100:.2f}%")

if __name__ == "__main__":
    train_vqa_model()
