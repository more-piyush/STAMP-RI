"""
Training utilities for DR augmentation pipeline.
Contains dataset, model, training loop, and evaluation functions.
Import with: from training_utils import *
"""

import os, copy, random
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.manifold import TSNE


# ===================== Dataset =====================

class ImprovedSupervisedTripletDataset(Dataset):
    """Triplet dataset with weighted sampling and stratified negative mining."""

    def __init__(self, base_dir, grade_folders, aug_suffix, aug_folder_suffix,
                 transform=None, neg_seed=42):
        self.transform = transform
        self.samples = []
        self.grade_originals = {}

        for grade_idx, grade in enumerate(grade_folders):
            orig_dir = os.path.join(base_dir, grade)
            aug_dir = os.path.join(base_dir, f'{grade}{aug_folder_suffix}')
            grade_orig_paths = []
            for f in sorted(os.listdir(orig_dir)):
                if f.startswith('c') and f.endswith('.jpg'):
                    base_name = os.path.splitext(f)[0]
                    aug_name = f'{base_name}{aug_suffix}'
                    aug_path = os.path.join(aug_dir, aug_name)
                    if os.path.exists(aug_path):
                        anchor_path = os.path.join(orig_dir, f)
                        self.samples.append((anchor_path, aug_path, grade_idx))
                        grade_orig_paths.append(anchor_path)
            self.grade_originals[grade_idx] = grade_orig_paths

        # Stratified negatives — round-robin across other grades
        rng = np.random.RandomState(neg_seed)
        self.neg_paths = []
        all_grades = sorted(self.grade_originals.keys())
        for _, _, grade_idx in self.samples:
            other_grades = [g for g in all_grades if g != grade_idx]
            neg_grade = other_grades[len(self.neg_paths) % len(other_grades)]
            neg_idx = rng.randint(0, len(self.grade_originals[neg_grade]))
            self.neg_paths.append(self.grade_originals[neg_grade][neg_idx])

        self.labels = [s[2] for s in self.samples]
        label_counts = Counter(self.labels)
        total = len(self.labels)
        self.sample_weights = [total / label_counts[label] for label in self.labels]
        print(f'  {len(self.samples)} triplets, class dist: {dict(sorted(label_counts.items()))}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, pos_path, _ = self.samples[idx]
        neg_path = self.neg_paths[idx]
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(pos_path).convert('RGB')
        negative = Image.open(neg_path).convert('RGB')
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        return anchor, positive, negative


# ===================== Model =====================

class EmbeddingNet(nn.Module):
    """ResNet-18 backbone with 128-dim L2-normalised output."""

    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        return F.normalize(self.resnet(x), p=2, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        return F.relu(pos_dist - neg_dist + self.margin).mean()


# ===================== Training Utilities =====================

def seed_everything(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_weighted_loader(dataset, batch_size=16, seed=42):
    """DataLoader with WeightedRandomSampler for class-balanced batches."""
    weights = torch.DoubleTensor(dataset.sample_weights)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True,
        generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )


def train_with_checkpoints(model, loader, optimizer, loss_fn,
                           total_epochs, checkpoint_every, device):
    """Train triplet model, saving state-dict checkpoints at intervals."""
    model.train()
    checkpoints = {}
    for epoch in range(1, total_epochs + 1):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}')
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}')
        if epoch % checkpoint_every == 0:
            checkpoints[epoch] = copy.deepcopy(model.state_dict())
            print(f'  >> Checkpoint saved at epoch {epoch}')
    return checkpoints


# ===================== Evaluation Utilities =====================

def extract_embeddings_with_labels(model, dataset, device):
    """Extract 128-dim embeddings for all anchors in the dataset."""
    model.eval()
    embeddings = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    with torch.no_grad():
        for anchor, _, _ in loader:
            emb = model(anchor.to(device))
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings), np.array(dataset.labels)


def linear_probe_accuracy(embeddings, labels, n_classes=5, epochs=100, lr=0.01):
    """5-fold stratified CV linear probe on frozen embeddings."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros_like(labels)
    for train_idx, val_idx in skf.split(embeddings, labels):
        X_train = torch.FloatTensor(embeddings[train_idx])
        y_train = torch.LongTensor(labels[train_idx])
        X_val = torch.FloatTensor(embeddings[val_idx])
        classifier = nn.Linear(128, n_classes)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        classifier.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(classifier(X_train), y_train)
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.no_grad():
            all_preds[val_idx] = classifier(X_val).argmax(dim=1).numpy()
    return (all_preds == labels).mean(), all_preds
