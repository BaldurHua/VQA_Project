import torch
import torch.nn.functional as F
from utils.model import SANModel, VocabDict
from torchvision import transforms
import cv2
import numpy as np

QST_VOCAB_PATH = "C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_questions.txt"
ANS_VOCAB_PATH = "C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_answers.txt"
MODEL_PATH = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/best_acc_model_v2.pt"
EMBED_SIZE = 512

qst_vocab = VocabDict(QST_VOCAB_PATH)
with open(ANS_VOCAB_PATH, "r", encoding="utf-8") as f:
    ans_vocab = [line.strip() for line in f]

model = SANModel(
    embed_size=EMBED_SIZE,
    qst_vocab=qst_vocab,
    ans_vocab_size=len(ans_vocab),
    word_embed_size=300,
    num_layers=2,
    hidden_size=256,
    freeze_emb=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

model.eval()

def encode_question(question: str, max_len=30):
    word2idx_dict = qst_vocab.word2idx_dict
    unk2idx = word2idx_dict.get("<unk>", 0)
    q_list = question.lower().strip().split()
    q_indices = [word2idx_dict.get(w, unk2idx) for w in q_list]
    padded = [unk2idx] * max_len
    padded[:len(q_indices)] = q_indices[:max_len]
    return torch.tensor(padded).long()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = torch.from_numpy(image).float() / 255.0
    image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    return image.unsqueeze(0)

def run_inference(image_tensor, question_tensor):
    image_tensor = image_tensor.to(device)
    question_tensor = question_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor, question_tensor)  
        probs = F.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(5, dim=1)
        return [(ans_vocab[top_indices[0][i].item()], top_probs[0][i].item()) for i in range(5)]