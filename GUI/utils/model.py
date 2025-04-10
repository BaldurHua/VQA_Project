import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
import cv2
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
            
        }

        if "question_type" in vqa:
            sample["question_type"] = vqa["question_type"]

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
    def __init__(self, qst_vocab, word_embed_size, embed_size, num_layers=2, hidden_size=256, freeze_emb=False):
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
        self.fc = nn.Linear(2 * hidden_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.tanh = nn.Tanh()

    def forward(self, question):
          qst_vec = self.word2vec(question)  
          qst_vec = self.embedding_dropout(qst_vec)
          qst_vec, (hidden, _) = self.lstm(qst_vec)  
          hidden = hidden[-2:].transpose(0, 1).contiguous().view(hidden.size(1), -1)

          qst_feature = self.tanh(self.fc(hidden))
          qst_feature = self.norm(qst_feature)
          if qst_feature.dim() == 4:
            qst_feature = qst_feature.squeeze(1)
          return qst_feature


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


class MLPBlock(nn.Module):
    def __init__(self, embed_size, ans_vocab_size, dropout=0.3):
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
        x= self.norm3(self.gelu(self.fc3(x)))
        return self.fc_out(x)

class SANModel(nn.Module):
    def __init__(self, embed_size, qst_vocab, ans_vocab_size, word_embed_size, num_layers, hidden_size, freeze_emb=True):
        super(SANModel, self).__init__()
        self.embed_size = embed_size
        self.num_attention_layer = 2

        # Image Encoder
        self.img_encoder = ResIncepEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab, word_embed_size, embed_size, num_layers, hidden_size, freeze_emb)
        self.san = nn.ModuleList([Attention(embed_size, dropout=0.2) for _ in range(self.num_attention_layer)])

        # MLP
        self.mlp = MLPBlock(embed_size, ans_vocab_size)
      
    def forward(self, img, qst):
        # Encode Image & Question
        img_feature = self.img_encoder(img)  
        qst_feature = self.qst_encoder(qst) 

        # Stacked Attention
        u = qst_feature
        for attn_layer in self.san:
            u = F.layer_norm(attn_layer(img_feature, u) + u, [self.embed_size])

        combined_feature = self.mlp(u)  
        return combined_feature
