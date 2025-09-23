"""
Multi-Stream Cross-Attention Synthesizer for Something-Something-v2 Action Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# We'll build this incrementally due to space constraints

class HandGNNEncoder(nn.Module):
    """Simplified Hand encoder using MLP for hand landmarks."""
    
    def __init__(self, embedding_dim=128):
        super(HandGNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Simple MLP for 21 landmarks * 2 coordinates = 42 features
        self.encoder = nn.Sequential(
            nn.Linear(42, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, hand_landmarks):
        # hand_landmarks: [batch_size, seq_len, 21, 2] or [batch_size, 21, 2]
        original_shape = hand_landmarks.shape
        
        if len(original_shape) == 4:  # Sequence input
            batch_size, seq_len = original_shape[:2]
            hand_landmarks = hand_landmarks.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 42]
            embeddings = self.encoder(hand_landmarks)  # [batch_size, seq_len, embedding_dim]
        else:  # Single frame
            hand_landmarks = hand_landmarks.view(hand_landmarks.shape[0], -1)  # [batch_size, 42]
            embeddings = self.encoder(hand_landmarks)  # [batch_size, embedding_dim]
            
        return embeddings

class ObjectCNNEncoder(nn.Module):
    """CNN encoder for object appearance using MobileNetV2."""
    
    def __init__(self, embedding_dim=256, pretrained=True):
        super(ObjectCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        self.backbone.classifier = nn.Identity()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )   
        
    def forward(self, object_crops):
        original_shape = object_crops.shape
        
        if len(original_shape) == 5:  # [batch_size, seq_len, 3, H, W]
            batch_size, seq_len = original_shape[:2]
            object_crops = object_crops.view(-1, *original_shape[2:])
        else:
            batch_size = original_shape[0]
            seq_len = 1
        
        features = self.backbone(object_crops)
        embeddings = self.projection(features)
        
        if len(original_shape) == 5:
            embeddings = embeddings.view(batch_size, seq_len, self.embedding_dim)
        
        return embeddings

class ContextCNNEncoder(nn.Module):
    """CNN encoder for scene context using ResNet34."""
    
    def __init__(self, embedding_dim=512, pretrained=True):
        super(ContextCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.backbone = models.resnet34(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, context_frames):
        original_shape = context_frames.shape
        
        if len(original_shape) == 5:
            batch_size, seq_len = original_shape[:2]
            context_frames = context_frames.view(-1, *original_shape[2:])
        else:
            batch_size = original_shape[0]
            seq_len = 1
        
        features = self.backbone(context_frames)
        embeddings = self.projection(features)
        
        if len(original_shape) == 5:
            embeddings = embeddings.view(batch_size, seq_len, self.embedding_dim)
        
        return embeddings

class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention for fusing multi-modal features."""
    
    def __init__(self, hand_dim=128, object_dim=256, context_dim=512, 
                 fusion_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.head_dim = fusion_dim // num_heads
        
        self.query_projection = nn.Linear(hand_dim, fusion_dim)
        combined_dim = object_dim + context_dim
        self.key_projection = nn.Linear(combined_dim, fusion_dim)
        self.value_projection = nn.Linear(combined_dim, fusion_dim)
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hand_embeddings, object_embeddings, context_embeddings):
        batch_size, seq_len = hand_embeddings.shape[:2]
        
        queries = self.query_projection(hand_embeddings)
        combined_features = torch.cat([object_embeddings, context_embeddings], dim=-1)
        keys = self.key_projection(combined_features)
        values = self.value_projection(combined_features)
        
        # Multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.fusion_dim
        )
        
        output = self.output_projection(attended_values)
        residual = self.query_projection(hand_embeddings)
        output = self.layer_norm(output + residual)
        
        return output

class TemporalAttention(nn.Module):
    """Temporal attention to focus on important time steps."""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, sequence):
        attention_weights = self.attention(sequence)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended_output = torch.sum(sequence * attention_weights, dim=1)
        return attended_output


class SynthesizerModel(nn.Module):
    """Complete Multi-Stream Cross-Attention Synthesizer model."""
    
    def __init__(self, num_classes=174, hand_dim=128, object_dim=256, 
                 context_dim=512, fusion_dim=512, lstm_hidden=256):
        super(SynthesizerModel, self).__init__()
        
        # Feature encoders
        self.hand_encoder = HandGNNEncoder(embedding_dim=hand_dim)
        self.object_encoder = ObjectCNNEncoder(embedding_dim=object_dim)
        self.context_encoder = ContextCNNEncoder(embedding_dim=context_dim)
        
        # Cross-attention fusion
        self.fusion_module = CrossAttentionFusion(
            hand_dim=hand_dim, object_dim=object_dim, 
            context_dim=context_dim, fusion_dim=fusion_dim
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(
            input_size=fusion_dim, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        
        self.temporal_attention = TemporalAttention(lstm_hidden * 2)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, num_classes)
        )
        
    def forward(self, hand_landmarks, object_crops, context_frames):
        # Stage 1: Parallel feature encoding
        hand_embeddings = self.hand_encoder(hand_landmarks)
        object_embeddings = self.object_encoder(object_crops)
        context_embeddings = self.context_encoder(context_frames)
        
        # Stage 2: Cross-attention fusion
        fused_embeddings = self.fusion_module(
            hand_embeddings, object_embeddings, context_embeddings
        )
        
        # Stage 3: Temporal sequence analysis
        lstm_output, _ = self.lstm(fused_embeddings)
        attended_output = self.temporal_attention(lstm_output)
        logits = self.classifier(attended_output)
        
        return logits


def create_synthesizer_model(num_classes=174):
    """Factory function to create SynthesizerModel."""
    return SynthesizerModel(num_classes=num_classes)
