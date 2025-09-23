"""
Multi-Stream Cross-Attention Synthesizer for Something-Something-v2 Action Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️  torch_geometric not available. Using MLP for hand encoder.")

import math

# We'll build this incrementally due to space constraints

class HandGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for hand landmarks that understands 
    the hand's skeletal structure and kinematic relationships.
    """
    
    def __init__(self, embedding_dim=128):
        super(HandGNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # if TORCH_GEOMETRIC_AVAILABLE:
            # GNN layers: 2D coordinates -> 64 -> embedding_dim
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, embedding_dim)
        
        # Define hand skeleton structure (MediaPipe 21-landmark hand model)
        # This encodes the physical structure of a human hand
        self.register_buffer('edge_index', torch.tensor([
            # Thumb (landmarks 0-4)
            [0, 1], [1, 2], [2, 3], [3, 4],
            # Index finger (landmarks 0, 5-8)  
            [0, 5], [5, 6], [6, 7], [7, 8],
            # Middle finger (landmarks 0, 9-12)
            [0, 9], [9, 10], [10, 11], [11, 12],
            # Ring finger (landmarks 0, 13-16)
            [0, 13], [13, 14], [14, 15], [15, 16],
            # Pinky finger (landmarks 0, 17-20)
            [0, 17], [17, 18], [18, 19], [19, 20],
            # Palm connections (optional but helpful for stability)
            [5, 9], [9, 13], [13, 17]
        ], dtype=torch.long).t().contiguous())
        
        print("✅ Using GNN-based hand encoder with skeletal structure")
        # else:
        #     # Fallback to MLP if torch_geometric not available
        #     self.mlp_encoder = nn.Sequential(
        #         nn.Linear(42, 128),  # 21 landmarks * 2 coords = 42
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(128, 256),
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(256, embedding_dim)
        #     )
        #     print("⚠️  Using MLP fallback for hand encoder")
        
    def forward(self, hand_landmarks):
        # hand_landmarks: [batch_size, seq_len, 21, 2]
        if not TORCH_GEOMETRIC_AVAILABLE:
            # Fallback to MLP
            original_shape = hand_landmarks.shape
            if len(original_shape) == 4:  # Sequence input
                batch_size, seq_len = original_shape[:2]
                hand_landmarks = hand_landmarks.view(batch_size, seq_len, -1)
                return self.mlp_encoder(hand_landmarks)
            else:  # Single frame
                hand_landmarks = hand_landmarks.view(hand_landmarks.shape[0], -1)
                return self.mlp_encoder(hand_landmarks)
        
        # GNN-based encoding
        batch_size, seq_len = hand_landmarks.shape[:2]
        
        # Reshape for GNN processing: [B, S, 21, 2] -> [B*S, 21, 2]
        node_features = hand_landmarks.view(-1, 21, 2)
        
        # Create batch vector for torch_geometric
        # Tells which nodes belong to which graph in the batch
        batch_vector = torch.arange(
            batch_size * seq_len, 
            device=hand_landmarks.device
        ).repeat_interleave(21)
        
        # GNN message passing
        # Each landmark updates its features based on connected landmarks
        x = node_features.view(-1, 2)  # [B*S*21, 2]
        x = F.relu(self.conv1(x, self.edge_index))  # [B*S*21, 64]
        x = self.conv2(x, self.edge_index)  # [B*S*21, embedding_dim]
        
        # Graph pooling: aggregate 21 landmark embeddings -> single hand embedding
        graph_embeddings = global_mean_pool(x, batch_vector)  # [B*S, embedding_dim]
        
        # Reshape back to sequence format: [B*S, embedding_dim] -> [B, S, embedding_dim]
        output = graph_embeddings.view(batch_size, seq_len, self.embedding_dim)
        
        return output

class ObjectCNNEncoder(nn.Module):
    """
    CNN encoder for object appearance using a pre-trained MobileNetV2.
    It takes a sequence of pre-cropped object images as input.
    """
    
    def __init__(self, embedding_dim=256, pretrained=True):
        super(ObjectCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 1. Load a pre-trained MobileNetV2 as the feature extraction backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # 2. Remove the final classification layer to get raw features (output size: 1280)
        self.backbone.classifier = nn.Identity()
        
        # 3. Create a "projection head" to map the raw features to our desired embedding size
        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )   
        
    def forward(self, object_crops):
        # Input shape: [batch_size, seq_len, 3, H, W]
        original_shape = object_crops.shape
        batch_size, seq_len = original_shape[:2]
        
        # Merge batch and sequence dims to process all crops at once: [B*S, C, H, W]
        object_crops_flat = object_crops.view(-1, *original_shape[2:])
        
        # Get features from the backbone: [B*S, 1280]
        features = self.backbone(object_crops_flat)
        
        # Project features to our final embedding dimension: [B*S, embedding_dim]
        embeddings = self.projection(features)
        
        # Reshape the output back to a sequence: [B, S, embedding_dim]
        output_embeddings = embeddings.view(batch_size, seq_len, self.embedding_dim)
        
        return output_embeddings

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
