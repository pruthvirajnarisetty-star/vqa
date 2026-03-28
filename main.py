import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import re
from PIL import Image
import torchvision.transforms as transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import numpy as np
import os

# Global models dictionary
models_dict = None
device = None

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_answers):
        super().__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(512 + hidden_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_answers)

    def forward(self, image, question):
        img_feat = self.cnn(image)
        q_embed = self.embedding(question)
        _, (h, _) = self.lstm(q_embed)
        q_feat = h.squeeze(0)
        combined = torch.cat((img_feat, q_feat), dim=1)
        x = self.relu(self.fc1(combined))
        out = self.fc2(x)
        return out

def load_models():
    """Load all models once at startup"""
    global models_dict, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load custom VQA model
    with open("models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("models/answer_mapping.pkl", "rb") as f:
        idx_to_answer = pickle.load(f)
    
    vocab_size = len(vocab)
    model = VQAModel(vocab_size, 300, 256, len(idx_to_answer))
    model.load_state_dict(torch.load("models/vqa_custom_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # BLIP2 for open-ended (smaller model for free tier)
    print("Loading BLIP2...")
    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-flan-t5-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-base",
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Translator
    print("Loading Translator...")
    translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    lang_code_map = {
        "en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu",
        "ta": "tam_Taml", "kn": "kan_Knda", "ml": "mal_Mlym"
    }
    
    models_dict = {
        'model': model, 'vocab': vocab, 'idx_to_answer': idx_to_answer,
        'processor': processor, 'blip_model': blip_model,
        'translator_tokenizer': translator_tokenizer,
        'translator_model': translator_model, 'lang_code_map': lang_code_map,
        'device': device
    }
    
    print("✅ All models loaded successfully!")
    return models_dict

def init_models():
    """Initialize models if not loaded"""
    global models_dict
    if models_dict is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        load_models()
    return models_dict

# All your functions remain EXACTLY the same...
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text

def encode_question_infer(q, vocab):
    q = clean_text(q)
    tokens = q.split()
    MAX_LEN = 20
    enc = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    enc = enc[:MAX_LEN] + [vocab["<pad>"]] * (MAX_LEN - len(enc))
    return torch.tensor(enc, dtype=torch.long)

def translate(text, src_lang, tgt_lang, tokenizer, model, lang_code_map, device):
    try:
        tokenizer.src_lang = lang_code_map.get(src_lang, "eng_Latn")
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(lang_code_map[tgt_lang]),
            max_length=50, num_beams=5
        )
        return tokenizer.decode(tokens[0], skip_special_tokens=True)
    except:
        return text

def predict_custom_vqa(image_tensor, question_tensor, model, idx_to_answer, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        question_tensor = question_tensor.to(device)
        out = model(image_tensor, question_tensor)
        _, pred = torch.max(out, 1)
    return idx_to_answer[pred.item()]

def open_vqa(image, question, processor, blip_model):
    inputs = processor(image, question, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(blip_model.device) for k, v in inputs.items()}
    out = blip_model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

def final_pipeline(image_path_or_pil, question):
    """Main inference function - EXACT SAME as before"""
    init_models()
    m = models_dict
    
    if hasattr(image_path_or_pil, 'convert'):
        image = image_path_or_pil.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
    else:
        image = Image.open(image_path_or_pil).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
    
    try:
        lang = detect(question)
    except:
        lang = "en"
    
    if lang != "en":
        q_en = translate(question, lang, "en", 
                        m['translator_tokenizer'], m['translator_model'], 
                        m['lang_code_map'], m['device'])
    else:
        q_en = question
    
    if any(x in q_en.lower() for x in ["what is", "describe", "this place", "show"]):
        answer_en = open_vqa(image, q_en, m['processor'], m['blip_model'])
    else:
        q_tensor = encode_question_infer(q_en, m['vocab']).unsqueeze(0)
        answer_en = predict_custom_vqa(image_tensor, q_tensor, 
                                     m['model'], m['idx_to_answer'], m['device'])
    
    if lang != "en":
        answer = translate(answer_en, "en", lang, 
                          m['translator_tokenizer'], m['translator_model'], 
                          m['lang_code_map'], m['device'])
    else:
        answer = answer_en
    
    return f"**Detected Language:** {lang}\n**Answer:** {answer}"
