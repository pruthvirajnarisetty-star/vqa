"""
Inference pipeline for VQA with multilingual support
"""
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from vqa_model import VQAModel

class VQAPipeline:
    def __init__(self, model_path: str, vocab_path: str, answer_map_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load custom model
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        with open(answer_map_path, "rb") as f:
            self.idx_to_answer = pickle.load(f)
        
        self.custom_model = VQAModel(len(self.vocab), 300, 256, len(self.idx_to_answer))
        self.custom_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.custom_model.to(self.device)
        self.custom_model.eval()
        
        # Load BLIP-2
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load translator
        self.translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        
        self.lang_code_map = {
            "en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu",
            "ta": "tam_Taml", "kn": "kan_Knda", "ml": "mal_Mlym"
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.MAX_LEN = 20
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src = self.lang_code_map.get(src_lang, "eng_Latn")
        tgt = self.lang_code_map.get(tgt_lang, "eng_Latn")
        inputs = self.translator_tokenizer(text, return_tensors="pt")
        generated_tokens = self.translator_model.generate(
            **inputs,
            forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[tgt]
        )
        return self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def encode_question(self, q: str) -> torch.Tensor:
        q = q.lower()
        tokens = q.split()
        enc = [self.vocab.get(w, self.vocab["<unk>"]) for w in tokens]
        enc = enc[:self.MAX_LEN] + [self.vocab["<pad>"]] * (self.MAX_LEN - len(enc))
        return torch.tensor(enc).unsqueeze(0)
    
    def predict_custom_vqa(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        q = self.encode_question(question).to(self.device)
        with torch.no_grad():
            out = self.custom_model(image, q)
            _, pred = torch.max(out, 1)
        return self.idx_to_answer[pred.item()]
    
    def open_vqa(self, image_path: str, question: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.blip_processor(image, question, return_tensors="pt").to(self.blip_model.device)
        out = self.blip_model.generate(**inputs, max_new_tokens=50)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)
    
    def predict(self, image_path: str, question: str) -> str:
        lang = detect(question)
        if lang != "en":
            q_en = self.translate(question, lang, "en")
        else:
            q_en = question
        
        if ("what is" in q_en.lower()) or ("this place" in q_en.lower()):
            answer_en = self.open_vqa(image_path, q_en)
        else:
            answer_en = self.predict_custom_vqa(image_path, q_en)
        
        if lang != "en":
            return self.translate(answer_en, "en", lang)
        else:
            return answer_en
