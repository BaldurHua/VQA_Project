import torch
import numpy as np
import cv2
import torch.nn.functional as F
from model import SANModel, VocabDict   

image_path = "C:/Users/Baldu/Desktop/Temp/VQA/data/images/test2015/COCO_test2015_000000000958.jpg"
question = "what does the sign say?"
saved_model_path = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/test_best_acc_model_0.2.pt"
vocab_q_path = "C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_questions.txt"
vocab_a_path = "C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_answers.txt"
max_qst_length = 30
embed_size = 756
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def encode_question(question, word2idx, unk_idx, max_len=30):
    q_list = question.lower().split()
    q_indices = [word2idx.get(w, unk_idx) for w in q_list]
    padded = [unk_idx] * max_len
    padded[:len(q_indices)] = q_indices[:max_len]
    return torch.tensor(padded).long()

qst_vocab = load_vocab(vocab_q_path)
ans_vocab = load_vocab(vocab_a_path)
word2idx_dict = {w: idx for idx, w in enumerate(qst_vocab)}
unk2idx = word2idx_dict.get("<unk>", 0)

q_encoded = encode_question(question, word2idx_dict, unk2idx, max_qst_length).unsqueeze(0).to(device)

image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = image[:, :, ::-1].transpose(2, 0, 1) 
image = torch.from_numpy(image).float() / 255.0
image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
image = image.unsqueeze(0).to(device)

model = SANModel(
    embed_size=embed_size,
    qst_vocab=VocabDict(vocab_q_path),
    ans_vocab_size=len(ans_vocab),
    word_embed_size=300,
    num_layers=2,
    hidden_size=256,
    freeze_emb=True
)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    output = model(image, q_encoded)
    probs = F.softmax(output, dim=1)
    top_probs, top_indices = probs.topk(5, dim=1)

print("\nTop 5 Predictions:")
for i in range(5):
    ans = ans_vocab[top_indices[0][i].item()]
    prob = top_probs[0][i].item()
    print(f"{ans:20s} - {prob:.4f}")