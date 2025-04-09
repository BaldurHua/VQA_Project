#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pickle
import os


# In[119]:


with open("C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/preprocessed_train.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/preprocessed_val.pkl", "rb") as f:
    val_data = pickle.load(f)

with open("C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/preprocessed_test.pkl", "rb") as f:
    test_data = pickle.load(f)

# print(f"Training Data Loaded: {len(train_data)} samples")
# print(f"Validation Data Loaded: {len(val_data)} samples")
# print(f"Test Data Loaded: {len(test_data)} samples")


# In[120]:


import torch
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
import cv2
import gc


# In[121]:


import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def tokenize(sentence):
    tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:

    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w:n_w for n_w, w in enumerate(self.word_list)}
        self.vocab_size = len(self.word_list)
        self.unk2idx = self.word2idx_dict[''] if '' in self.word2idx_dict else None

    def idx2word(self, n_w):

        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            return self.word2idx_dict.get("<unk>", 0) 

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]

        return inds


# In[125]:


class VqaDataset(Dataset):
    def __init__(self, input_dir, input_vqa, max_qst_length=30, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(f"{input_dir}/{input_vqa}", allow_pickle=True)
        self.qst_vocab = VocabDict(f"{input_dir}/vocab_questions.txt")
        self.ans_vocab = VocabDict(f"{input_dir}/vocab_answers.txt")
        self.max_qst_length = max_qst_length
        self.transform = transform
        self.load_ans = "answer_label" in self.vqa[0]

    def __getitem__(self, idx):
        vqa = self.vqa[idx]

        # Load Image
        image = cv2.imread(vqa["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Process Question 
        qst_tokens = vqa["question_tokens"]
        qst2idc = np.full(self.max_qst_length, self.qst_vocab.word2idx("<unk>"))
        qst2idc[:len(qst_tokens)] = qst_tokens[:self.max_qst_length]

        sample = {
            "image": image,
            "question": torch.tensor(qst2idc, dtype=torch.long),
            # "question_type": vqa["question_type"]
        }

        # Process Answers
        if self.load_ans:
            sample["answer_label"] = torch.tensor(vqa["answer_label"], dtype=torch.long)

            MAX_ANSWERS = 10
            answer_multi_choice = vqa.get("valid_answers", [])

            if len(answer_multi_choice) < MAX_ANSWERS:
                # answer_multi_choice += [0] * (MAX_ANSWERS - len(answer_multi_choice))
                answer_multi_choice += [-1] * (MAX_ANSWERS - len(answer_multi_choice))
            else:
                answer_multi_choice = answer_multi_choice[:MAX_ANSWERS]

            sample["answer_multi_choice"] = torch.tensor(answer_multi_choice, dtype=torch.long)

        return sample

    def __len__(self):
        return len(self.vqa)

    


# In[126]:

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  
])

train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_train.pkl", 
                           transform=train_transform)

val_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_val.pkl", 
                           transform=val_transform)

test_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_test.pkl", 
                           transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
# test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, pin_memory=True)

# sample_batch = next(iter(train_loader))
# print(sample_batch["image"].shape) 
# print(sample_batch["question"].shape) 
# print(sample_batch.get("answer_label", None))

# In[]:
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import random

# Load a random sample
# idx = random.randint(0, len(train_dataset) - 1)
# raw_path = train_dataset.vqa[idx]["image_path"]
# raw_img = Image.open(raw_path).convert("RGB")
# aug_img = train_transform(raw_img)

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(raw_img)
# axs[0].set_title("Original Image")
# axs[0].axis('off')

# aug_img_disp = transforms.ToPILImage()(aug_img)
# axs[1].imshow(aug_img_disp)
# axs[1].set_title("Transformed Image")
# axs[1].axis('off')

# plt.tight_layout()
# plt.savefig("figure1_preprocessing.png")
# plt.show()

# In[ ]:
# idx = random.randint(0, len(train_dataset) - 1)
# vqa = train_dataset.vqa[idx]
# token_ids = vqa["question_tokens"]
# tokens = [train_dataset.qst_vocab.idx2word(i) for i in token_ids if i != 0] 
# question = " ".join(tokens)
# valid_answers = vqa["valid_answers"]

# token_ids = [train_dataset.qst_vocab.word2idx(w) for w in tokens]
# mapped_answers = [train_dataset.ans_vocab.word2idx(ans) for ans in valid_answers]

# print(f"Q: {question}")
# print(f"Tokens: {tokens}")
# print(f"Token IDs: {token_ids}")
# print(f"Valid Answers: {valid_answers}")
# print(f"Mapped IDs: {mapped_answers}")

# In[128]:
import torch
import torch.nn as nn
import torch.nn.functional as F

# def init_weights(m):
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

def init_weights(m):
    # Skip pretrained
    if hasattr(m, "weight") and not m.weight.requires_grad:
        return

    if isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        if fan_in < 2048:
            nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        out_channels_branch = 64 # tune  
        self.branch1x1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  
        branch_pool = self.branch_pool(branch_pool)  

        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  
        out = self.relu(out)

        return out


# In[129]:
class ResIncepEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ResIncepEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Blocks
        self.res_block1 = ResBlock(32)
        self.res_block2 = ResBlock(32)

        # Inception module
        self.inception = InceptionModule(32)  
        # self.inception_bn = nn.BatchNorm2d(64)
        self.inception_bn = nn.BatchNorm2d(256)

        # Double Convolution Layers
        # self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(inplace=True)
        
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = nn.ReLU(inplace=True)
        
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Feature Embedding Layer
        self.fc_embed = nn.Linear(128, embed_size)

        self.norm = nn.LayerNorm(embed_size)
        
        # Initialize Weights
        self.apply(init_weights)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.inception(x)

        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))

        x = self.pool2(x)  

        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width).transpose(1, 2)  

        x = self.fc_embed(x)  
        x = self.norm(x)

        return x
    

# import torchvision.models as models
# from torchvision.models import ResNet50_Weights

# # Simpler approach
# class ResNetEncoder(nn.Module):
#     def __init__(self, embed_size):
#         super(ResNetEncoder, self).__init__()
#         self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         self.resnet.fc = nn.Identity()  
#         self.fc = nn.Linear(2048, embed_size)  
#         self.norm = nn.LayerNorm(embed_size)

#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.fc(x)  
#         x = self.norm(x)  
#         return x


#  In[130]:
import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-300") 

def build_embedding_matrix(qst_vocab, word_embed_size):
    vocab_size = qst_vocab.vocab_size
    embedding_matrix = torch.empty((vocab_size, word_embed_size), dtype=torch.float)

    # Initialize all embeddings 
    torch.nn.init.xavier_uniform_(embedding_matrix)

    # Override with GloVe vectors 
    for word, idx in qst_vocab.word2idx_dict.items():
        if word in glove_model.key_to_index:
            embedding_matrix[idx] = torch.tensor(glove_model[word], dtype=torch.float)

    # Set padding & unknown tokens to zero
    for word in ["<pad>", "<unk>"]:
        if word in qst_vocab.word2idx_dict:
            embedding_matrix[qst_vocab.word2idx_dict[word]] = torch.zeros(word_embed_size)

    return embedding_matrix

class QstEncoder(nn.Module):
    def __init__(self, qst_vocab, word_embed_size, embed_size, num_layers=2, hidden_size=256, freeze_emb=True):
        super(QstEncoder, self).__init__()

        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # pass the precomputed embedding matrix
        embedding_matrix = build_embedding_matrix(qst_vocab, word_embed_size)

        # Embedding layer 
        self.word2vec = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_emb)
        print(f"Initialized embedding layer with GloVe (freeze={freeze_emb})")

        self.embedding_dropout = nn.Dropout(0.2)

        # LSTM Encoder
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        # self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size) 
        self.fc = nn.Linear(2 * hidden_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.tanh = nn.Tanh()

    def forward(self, question):
          qst_vec = self.word2vec(question)  
          qst_vec = self.embedding_dropout(qst_vec)
          qst_vec, (hidden, _) = self.lstm(qst_vec)  
        #  hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
          hidden = hidden[-2:].transpose(0, 1).contiguous().view(hidden.size(1), -1)

          qst_feature = self.tanh(self.fc(hidden))
        #   qst_feature = self.tanh(self.fc(qst_vec))   
          qst_feature = self.norm(qst_feature)
          if qst_feature.dim() == 4:
            qst_feature = qst_feature.squeeze(1)
          return qst_feature



# class QstEncoder(nn.Module):
#     def __init__(self, qst_vocab, word_embed_size, embed_size, num_layers=2, hidden_size=256, freeze_emb=True):
#         super(QstEncoder, self).__init__()

#         self.word_embed_size = word_embed_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # Pretrained GloVe embeddings instead of randomized the embedding
#         vocab_size = qst_vocab.vocab_size
#         embedding_matrix = torch.zeros((vocab_size, word_embed_size), dtype=torch.float)

#         for word, idx in qst_vocab.word2idx_dict.items(): 
#             # if idx % 500 == 0: 
#             #     print(f"Processing word {idx}/{qst_vocab.vocab_size}: {word}")

#             if word in glove_model.key_to_index:
#                 embedding_matrix[idx] = torch.tensor(glove_model[word], dtype=torch.float)
#             else:
#                 embedding_matrix[idx] = torch.randn(word_embed_size) * 0.1

#         for word in ["<pad>", "<unk>"]:
#             if word in qst_vocab.word2idx_dict:
#                 idx = qst_vocab.word2idx_dict[word]
#                 embedding_matrix[idx] = torch.zeros(word_embed_size)  # Assign zero embeddings for padding
        
#         # missing_words = [word for word in qst_vocab.word_list if word.lower() not in glove_model.key_to_index]
#         # print(f"Missing words: {missing_words[:10]} (Total missing: {len(missing_words)})")

#         # Embedding layer
#         self.word2vec = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_emb)
#         print(f"Initialized embedding layer with GloVe (freeze={freeze_emb})")

#         # LSTM 
#         self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size) 
#         self.norm = nn.LayerNorm(embed_size)
#         self.tanh = nn.Tanh()

#     def forward(self, question):
#         qst_vec = self.word2vec(question)  
#         qst_vec, (hidden, _) = self.lstm(qst_vec)  

#         hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)  

#         qst_feature = self.tanh(self.fc(hidden))  
#         qst_feature = self.norm(qst_feature)
#         return qst_feature


# In[131]:

# class Attention(nn.Module):
#     def __init__(self, num_channels, embed_size, dropout=True):
#         super(Attention, self).__init__()
#         self.ff_image = nn.Linear(embed_size, num_channels)
#         self.ff_questions = nn.Linear(embed_size, num_channels)
#         self.dropout = nn.Dropout(p=0.5) if dropout else nn.Identity()
#         self.ff_attention = nn.Linear(num_channels, 1)
#         self.norm = nn.LayerNorm(embed_size)  

#     def forward(self, vi, vq):
#         hi = self.ff_image(vi)  
#         hq = self.ff_questions(vq).unsqueeze(dim=1)  
#         ha = torch.tanh(hi + hq)

#         ha = self.dropout(ha)  
#         ha = self.ff_attention(ha) 

#         pi = torch.softmax(ha, dim=1)  
#         vi_attended = (pi * vi).sum(dim=1)

#         u = self.norm(vi_attended + vq) 
#         return u

class Attention(nn.Module):
    def __init__(self, embed_size, dropout=True):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        
        # Linear layers 
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(p=0.3) if dropout else nn.Identity()
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, vi, vq):
        q = self.q_linear(vq).unsqueeze(1) 
        k = self.k_linear(vi)  
        v = self.v_linear(vi) 

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_size ** 0.5)  
        attn_weights = torch.softmax(attn_scores, dim=-1)  

        vi_attended = torch.matmul(attn_weights, v).squeeze(1) 

        u = self.norm(vi_attended + vq)  

        return u

# Simplified attention module
# class Attention(nn.Module):
#     def __init__(self, embed_size, dropout=0.4):
#         super(Attention, self).__init__()
#         self.img_proj = nn.Linear(embed_size, embed_size)
#         self.q_proj = nn.Linear(embed_size, embed_size)
#         self.attn = nn.Linear(embed_size, 1)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(embed_size)

#     def forward(self, vi, vq):
#         hi = self.img_proj(vi)  
#         hq = self.q_proj(vq).unsqueeze(1)  
#         ha = torch.tanh(hi + hq)
#         ha = self.dropout(self.attn(ha))  
#         pi = torch.softmax(ha, dim=1)  
#         vi_attended = (pi * vi).sum(dim=1)  
#         return self.norm(vi_attended + vq)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_size, num_heads=2, dropout=0.5):
#         super(MultiHeadAttention, self).__init__()
#         self.mha = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(embed_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, vi, vq):
#         if vq.dim() == 4:  
#             print(f"Fixing shape: Original vq {vq.shape}")
#             vq = vq.squeeze(1) 
#         attn_output, _ = self.mha(vq, vi, vi) 
#         u = self.norm(vq + attn_output).squeeze(1)  
#         return u



class MLPBlock(nn.Module):
    def __init__(self, embed_size, ans_vocab_size, dropout=0.4):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_size, 1024)
        self.gelu = nn.GELU()
        self.norm1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(1024, 512)
        self.norm2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        
        self.fc_out = nn.Linear(256, ans_vocab_size)

    def forward(self, x):
        x = self.dropout1(self.norm1(self.gelu(self.fc1(x))))
        x = self.dropout2(self.norm2(self.gelu(self.fc2(x))))
        # x = self.dropout2(self.norm3(self.gelu(self.fc3(x))))
        x= self.norm3(self.gelu(self.fc3(x)))
        return self.fc_out(x)
    
# Simplified MLP block
# class MLPBlock(nn.Module):
#     def __init__(self, embed_size, ans_vocab_size, dropout=0.3):
#         super(MLPBlock, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_size, 1024),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(1024, ans_vocab_size)
#         )

#     def forward(self, x):
#         return self.mlp(x)




# In[132]:

class SANModel(nn.Module):
    def __init__(self, embed_size, qst_vocab, ans_vocab_size, word_embed_size, num_layers, hidden_size, freeze_emb=True):
        super(SANModel, self).__init__()
        self.embed_size = embed_size
        self.num_attention_layer = 2

        # Image Encoder
        self.img_encoder = ResIncepEncoder(embed_size)
        # self.img_encoder = ResNetEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab, word_embed_size, embed_size, num_layers, hidden_size, freeze_emb)
        # self.san = nn.ModuleList([MultiHeadAttention(embed_size, embed_size) for _ in range(self.num_attention_layer)])
        self.san = nn.ModuleList([Attention(embed_size, dropout=0.3) for _ in range(self.num_attention_layer)])

        # MLP
        self.mlp = MLPBlock(embed_size, ans_vocab_size)
      
    def forward(self, img, qst):
        # Encode Image & Question
        img_feature = self.img_encoder(img)  
        qst_feature = self.qst_encoder(qst) 

        # Stacked Attention
        u = qst_feature
        for attn_layer in self.san:
            # u = attn_layer(img_feature, u)
            # u = attn_layer(img_feature, u) + u
            u = F.layer_norm(attn_layer(img_feature, u) + u, [self.embed_size])

        # Pooling for mha
        # u = u.mean(dim=1)  

        combined_feature = self.mlp(u)  
        return combined_feature


# In[ ]:
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR,  SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Subset
import torch.optim as optim
import time

def train():
    # Clear GPU cache
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_train.pkl", 
                           transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True, persistent_workers=True)

    # ratio = 0.3  
    # subset_size = int(len(train_dataset) * ratio)  
    # indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    # train_subdataset = Subset(train_dataset, indices)
    # train_loader = DataLoader(train_subdataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_val.pkl", 
                           transform=val_transform)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

    model_dir = 'C:/Users/Baldu/Desktop/Temp/VQA/outputs'
    log_dir = 'C:/Users/Baldu/Desktop/Temp/VQA/outputs'

    train_dataset = train_loader.dataset

    qst_vocab_size = train_dataset.qst_vocab.vocab_size
    ans_vocab_size = train_dataset.ans_vocab.vocab_size
    ans_unk_idx = train_dataset.ans_vocab.unk2idx
    # qst_vocab_size = train_subdataset.dataset.qst_vocab.vocab_size
    # ans_vocab_size = train_subdataset.dataset.ans_vocab.vocab_size
    # ans_unk_idx = train_subdataset.dataset.ans_vocab.unk2idx

    use_saved_model = True
    checkpoint_path = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/checkpoint-epoch-15.pth" 
    # checkpoint_path = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/test_checkpoint-epoch-30.pth" 
    best_model_path = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/best_model.pt"
    best_acc_model_path = "C:/Users/Baldu/Desktop/Temp/VQA/outputs/best_acc_model.pt" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = SANModel(
        embed_size=512, 
        qst_vocab=VocabDict('C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_questions.txt'),
        ans_vocab_size=ans_vocab_size,
        word_embed_size=300,
        num_layers=2,
        hidden_size=256,
        freeze_emb=False
    )

    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)  
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=5e-5)
    # scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5)

    # warmup_scheduler = LinearLR(optimizer, start_factor=0.2, total_iters=3)  
    # main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-5)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[3])

    if use_saved_model:
        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])  
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
                start_epoch = checkpoint['epoch']  
                prev_loss = checkpoint.get('loss', None)  
                print(f"Resumed from epoch {start_epoch}, previous loss: {prev_loss:.4f}" if prev_loss else f"Resumed from epoch {start_epoch}")
            else:
                print("Loading saved best model.")
                model.load_state_dict(torch.load(best_acc_model_path, map_location=device))
                start_epoch = 0  
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
            start_epoch = 0
            model.apply(init_weights)  
    else:
        print("Starting new model.")
        start_epoch = 0
        model.apply(init_weights)

    last_time = 0
    early_stop_threshold = 3
    best_loss = float("inf")
    best_acc = 0.0
    val_increase_count = 0
    stop_training = False
    prev_loss = float("inf")
    num_epochs = 5
    save_step = 5

    print("Starting training...")
    for epoch in range(start_epoch, start_epoch + num_epochs):
            if stop_training:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            for phase in ['train', 'valid']: 
                dataloader = train_loader if phase == 'train' else val_loader
                dataset_size = len(dataloader.dataset)

                running_loss = 0.0
                running_corr = 0

                # soft accuracy
                running_soft_acc = 0
                valid_soft_count = 0

                if phase == 'train':
                    model.train()
                else:
                    model.eval()


                last_time = time.time()

                for batch_idx, batch_sample in enumerate(dataloader): 
                    image = batch_sample['image'].to(device)
                    question = batch_sample['question'].to(device)
                    label = batch_sample['answer_label'].to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        with torch.no_grad() if phase == 'valid' else torch.enable_grad():   
                            output = model(image, question)

                            _, pred = torch.max(output, 1)
                            

                            # soft accuracy metrics
                            multi_choice = batch_sample['answer_multi_choice'].to(device)
                            # print("Pred:", pred[0].item())
                            # print("Valid Answers:", multi_choice[0].tolist())
                            valid_mask = (multi_choice != -1)
                            matches = (pred.unsqueeze(1) == multi_choice) & valid_mask
                            soft_scores = torch.clamp(matches.sum(dim=1).float() / 3.0, max=1.0)
                            # running_soft_acc += soft_scores.sum().item()

                            valid_rows = valid_mask.any(dim=1) 
                            running_soft_acc += soft_scores[valid_rows].sum().item()
                            valid_soft_count += valid_rows.sum().item()

                            loss = criterion(output, label)

                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()


                    correct_predictions = (pred == label).sum().item()
                    running_corr += correct_predictions
                    running_loss += loss.item() * image.size(0)
                    
                    # Print periodically for progress
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        elapsed_time = time.time() - last_time
                        estimated_time = elapsed_time * (len(dataloader) - batch_idx) / 100
                        print(f'| {phase.upper()} | Epoch [{epoch+1}/{start_epoch + num_epochs}], '
                              f'Batch [{batch_idx}/{len(dataloader)}], '
                              f'Loss: {loss.item():.4f}, '
                              f'Estimated time left: {estimated_time/3600:.2f} hr')
                        last_time = time.time()

                epoch_loss = running_loss / dataset_size
                epoch_acc = running_corr / dataset_size

                # soft accuracy
                # epoch_soft_acc = running_soft_acc / dataset_size
                epoch_soft_acc = running_soft_acc / valid_soft_count

                print(f'| {phase.upper()} | Epoch [{epoch+1}/{start_epoch + num_epochs}], '
                      f'Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Soft Acc: {epoch_soft_acc:.4f}\n')
                
                print(f"Skipped {dataset_size - valid_soft_count} samples with all -1 answers.")


                # Logging
                log_file = os.path.join(log_dir, f'{phase}-log.txt')
                with open(log_file, 'a') as f:
                    f.write(f'Epoch {epoch+1}\tLoss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}\tSoft Acc: {epoch_soft_acc:.4f}\n')
                
                if phase == 'train':  
                    scheduler.step()

                # Validation
                if phase == 'valid':
                    # scheduler.step(epoch_loss)

                    for param_group in optimizer.param_groups:
                        print(f"Current Learning Rate: {param_group['lr']:.6f}")

                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'test_best_model.pt'))
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'test_best_model_0.2.pt'))
                        torch.save(model.state_dict(), os.path.join(model_dir, 'best_model_v2.pt'))

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'best_acc_model.pt'))
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'test_best_acc_model.pt'))
                        # torch.save(model.state_dict(), os.path.join(model_dir, 'test_best_acc_model_0.2.pt'))
                        torch.save(model.state_dict(), os.path.join(model_dir, 'best_acc_model_v2.pt'))

                    if epoch_loss > prev_loss:
                        val_increase_count += 1
                    else:
                        val_increase_count = 0

                    if val_increase_count >= early_stop_threshold:
                        stop_training = True

                    prev_loss = epoch_loss

            # scheduler.step()

            if (epoch + 1) % save_step == 0:
                torch.save({'epoch': epoch + 1,  
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_loss
                        }, os.path.join(model_dir, f'checkpoint-epoch-{epoch+1:02d}.pth'))
                print(f"Checkpoint saved at epoch {epoch + 1}")

            # if (epoch + 1) % save_step == 0:
            #     torch.save({
            #         'epoch': epoch + 1,  
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': epoch_loss
            #     }, os.path.join(model_dir, f'test_checkpoint-epoch-{epoch+1:02d}.pth'))
            #     print(f"Test checkpoint saved at epoch {epoch + 1}")
            

# In[ ]:
if __name__ == '__main__':
    train()

# In[ ]:
from torch.utils.data import DataLoader
from collections import defaultdict

test_dataset = VqaDataset(
    input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
    input_vqa="preprocessed_test.pkl", 
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('C:/Users/Baldu/Desktop/Temp/VQA/outputs/best_model_v2.pt', map_location=device)

def test_eval():
    model.to(device)
    model.eval()

    total_soft_acc = 0
    total_hard_acc = 0
    valid_soft_count = 0
    dataset_size = len(test_loader.dataset)

    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)

    with torch.no_grad():
        for batch_sample in test_loader:
            image = batch_sample['image'].to(device)
            question = batch_sample['question'].to(device)
            label = batch_sample['answer_label'].to(device)
            multi_choice = batch_sample['answer_multi_choice'].to(device)
            question_types = batch_sample['question_type'] 

            output = model(image, question)
            _, pred = torch.max(output, 1)

            # Hard accuracy
            total_hard_acc += (pred == label).sum().item()

            # Soft accuracy
            valid_mask = (multi_choice != -1)
            matches = (pred.unsqueeze(1) == multi_choice) & valid_mask
            soft_scores = torch.clamp(matches.sum(dim=1).float() / 3.0, max=1.0)
            valid_rows = valid_mask.any(dim=1)
            total_soft_acc += soft_scores[valid_rows].sum().item()
            valid_soft_count += valid_rows.sum().item()

            # Accuracy by question type
            for i, q_type in enumerate(question_types):
                total_by_type[q_type] += 1
                if pred[i].item() == label[i].item():
                    correct_by_type[q_type] += 1

    avg_soft_acc = total_soft_acc / valid_soft_count
    avg_acc = total_hard_acc / dataset_size

    print(f"\nStrict Accuracy (Test Set): {avg_acc:.4f}")
    print(f"Soft Accuracy (Test Set):   {avg_soft_acc:.4f}")

    print("\nAccuracy by Question Type:")
    for q_type in sorted(total_by_type.keys()):
        acc = 100.0 * correct_by_type[q_type] / total_by_type[q_type]
        print(f"{q_type.capitalize():<12}: {acc:.2f}%")

