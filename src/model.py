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
        }

        # Process Answers
        if self.load_ans:
            sample["answer_label"] = torch.tensor(vqa["answer_label"], dtype=torch.long)

            # Multi-choice 
            MAX_ANSWERS = 10
            answer_multi_choice = vqa.get("valid_answers", [])
            
            if len(answer_multi_choice) < MAX_ANSWERS:
                answer_multi_choice += [0] * (MAX_ANSWERS - len(answer_multi_choice))
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

train_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_train.pkl", 
                           transform=transform)

val_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_val.pkl", 
                           transform=transform)

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
    if isinstance(m, nn.Linear):
        if isinstance(m, nn.Linear) and m.weight.shape[0] < 2048:
            nn.init.xavier_uniform_(m.weight)  
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        out_channels_branch = 16 
        self.branch1x1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(out_channels_branch, out_channels_branch, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, out_channels_branch, kernel_size=1)
        
        self.apply(init_weights)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1) 
    
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        self.apply(init_weights)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

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

        # Double Convolution Layers
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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

        return x


# In[130]:
import gensim.downloader as api

glove_model = api.load("glove-wiki-gigaword-300") 

class QstEncoder(nn.Module):
    def __init__(self, qst_vocab, word_embed_size, embed_size, num_layers=2, hidden_size=256, freeze_emb=True):
        super(QstEncoder, self).__init__()

        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Pretrained GloVe embeddings instead of randomized the embedding
        vocab_size = qst_vocab.vocab_size
        embedding_matrix = torch.zeros((vocab_size, word_embed_size), dtype=torch.float)

        for word, idx in qst_vocab.word2idx_dict.items(): 
            # if idx % 500 == 0: 
            #     print(f"Processing word {idx}/{qst_vocab.vocab_size}: {word}")

            if word in glove_model.key_to_index:
                embedding_matrix[idx] = torch.tensor(glove_model[word], dtype=torch.float)
            else:
                embedding_matrix[idx] = torch.randn(word_embed_size) * 0.1

        for word in ["<pad>", "<unk>"]:
            if word in qst_vocab.word2idx_dict:
                idx = qst_vocab.word2idx_dict[word]
                embedding_matrix[idx] = torch.zeros(word_embed_size)  # Assign zero embeddings for padding
 
        
        # missing_words = [word for word in qst_vocab.word_list if word.lower() not in glove_model.key_to_index]
        # print(f"Missing words: {missing_words[:10]} (Total missing: {len(missing_words)})")

        # Embedding layer
        self.word2vec = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze_emb)
        print(f"Initialized embedding layer with GloVe (freeze={freeze_emb})")

        # LSTM 
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size) 
        self.norm = nn.LayerNorm(embed_size)
        self.tanh = nn.Tanh()

    def forward(self, question):
        qst_vec = self.word2vec(question)  
        qst_vec, (hidden, _) = self.lstm(qst_vec)  

        hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)  

        qst_feature = self.tanh(self.fc(hidden))  
        qst_feature = self.norm(qst_feature)
        return qst_feature


# In[131]:

class Attention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        super(Attention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.dropout = nn.Dropout(p=0.5) if dropout else nn.Identity()
        self.ff_attention = nn.Linear(num_channels, 1)
        self.norm = nn.LayerNorm(embed_size)  

    def forward(self, vi, vq):
        hi = self.ff_image(vi)  
        hq = self.ff_questions(vq).unsqueeze(dim=1)  
        ha = torch.tanh(hi + hq)

        ha = self.dropout(ha)  
        ha = self.ff_attention(ha) 

        pi = torch.softmax(ha, dim=1)  
        vi_attended = (pi * vi).sum(dim=1)

        u = self.norm(vi_attended + vq) 
        return u


# In[132]:

class SANModel(nn.Module):
    def __init__(self, embed_size, qst_vocab, ans_vocab_size, word_embed_size, num_layers, hidden_size, freeze_emb=True):
        super(SANModel, self).__init__()
        self.num_attention_layer = 4

        # Image Encoder
        self.img_encoder = ResIncepEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab, word_embed_size, embed_size, num_layers, hidden_size, freeze_emb)
        self.san = nn.ModuleList([Attention(embed_size, embed_size) for _ in range(self.num_attention_layer)])

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, ans_vocab_size) 
        )

    def forward(self, img, qst):
        # Encode Image & Question
        img_feature = self.img_encoder(img)  
        qst_feature = self.qst_encoder(qst) 

        # Stacked Attention
        u = qst_feature
        for attn_layer in self.san:
            u = attn_layer(img_feature, u)

        combined_feature = self.mlp(u)  
        return combined_feature



# In[ ]:
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
import torch.optim as optim
import time

def train():
    train_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_train.pkl", 
                           transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    ratio = 0.3  
    subset_size = int(len(train_dataset) * ratio)  
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subdataset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subdataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    val_dataset = VqaDataset(input_dir="C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed", 
                           input_vqa="preprocessed_val.pkl", 
                           transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    model_dir = 'C:/Users/Baldu/Desktop/Temp/VQA/outputs'
    log_dir = 'C:/Users/Baldu/Desktop/Temp/VQA/outputs'

    # train_dataset = train_loader.dataset

    # qst_vocab_size = train_dataset.qst_vocab.vocab_size
    # ans_vocab_size = train_dataset.ans_vocab.vocab_size
    # ans_unk_idx = train_dataset.ans_vocab.unk2idx
    qst_vocab_size = train_subdataset.dataset.qst_vocab.vocab_size
    ans_vocab_size = train_subdataset.dataset.ans_vocab.vocab_size
    ans_unk_idx = train_subdataset.dataset.ans_vocab.unk2idx


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SANModel(
            embed_size=512, # tune
            qst_vocab=VocabDict('C:/Users/Baldu/Desktop/Temp/VQA/data/preprocessed/vocab_questions.txt'),
            ans_vocab_size=ans_vocab_size,
            word_embed_size=300,
            num_layers=2,
            hidden_size=64)
    
    model.apply(init_weights)

    model.to(device)
    # print("Sample Embedding Weights:", model.qst_encoder.word2vec.weight[:5, :10])
    
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    last_time = 0
    early_stop_threshold = 3
    best_loss = 99999
    val_increase_count = 0
    stop_training = False
    prev_loss = 9999
    num_epochs = 10
    save_step = 1

    print("Starting training...")
    for epoch in range(num_epochs):
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        for phase in ['train', 'valid']: 
            dataloader = train_loader if phase == 'train' else val_loader
            dataset_size = len(dataloader.dataset)

            running_loss = 0.0
            running_corr = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()


            last_time = time.time()

            for batch_idx, batch_sample in enumerate(dataloader): 
                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                # multi_choice = batch_sample['answer_multi_choice']

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'): 
                    output = model(image, question)
                    # print("Model Output Shape:", output.shape)
                    # print("Label Shape:", label.shape)
                    # print("Output Sample:", output[:10])
                    # print("Prediction:", output.argmax(dim=1)[:10])

                    _, pred = torch.max(output, 1)
                    # print("Predictions vs Labels:", pred[:10], label[:10])
                    loss = criterion(output, label)
                
                    if phase == 'train':
                        loss.backward()
                        # for name, param in model.named_parameters():
                        #     if param.grad is not None:
                        #         print(f"{name}: grad mean={param.grad.abs().mean():.6f}, grad max={param.grad.abs().max():.6f}")
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Accuracy handling
                # if len(multi_choice) == 0:
                #     multi_choice = torch.zeros((pred.size(0), 1), dtype=torch.long).to(device)
                # else:
                #     multi_choice = multi_choice.clone().detach().to(device)

                # if multi_choice.dim() == 1:
                #     multi_choice = multi_choice.unsqueeze(0)

                correct_predictions = (pred == label).sum().item()
                running_corr += correct_predictions
                running_loss += loss.item() * image.size(0)

                # Print periodically for progress
                if batch_idx % 100 == 0 and batch_idx > 0:
                    elapsed_time = time.time() - last_time
                    estimated_time = elapsed_time * (len(dataloader) - batch_idx) / 100
                    print(f'| {phase.upper()} | Epoch [{epoch+1}/{num_epochs}], '
                          f'Batch [{batch_idx}/{len(dataloader)}], '
                          f'Loss: {loss.item():.4f}, '
                          f'Estimated time left: {estimated_time/3600:.2f} hr')
                    last_time = time.time()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corr / dataset_size

            print(f'| {phase.upper()} | Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}\n')

            # Logging
            log_file = os.path.join(log_dir, f'{phase}-log.txt')
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch+1}\tLoss: {epoch_loss:.4f}\tAcc: {epoch_acc:.4f}\n')

            # Validation
            if phase == 'valid':
                scheduler.step(epoch_loss)  

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))

                if epoch_loss > prev_loss:
                    val_increase_count += 1
                else:
                    val_increase_count = 0

                if val_increase_count >= early_stop_threshold:
                    stop_training = True

                prev_loss = epoch_loss

        if (epoch + 1) % save_step == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model-epoch-{epoch+1:02d}.pt'))

# In[ ]:
if __name__ == '__main__':
    train()




