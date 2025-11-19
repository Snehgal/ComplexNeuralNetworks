import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def segformer_SAS_2channel(num_class = 9):
    num_classes = num_class
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"

    # Load pretrained encoder + mismatched decoder (we will reinit decoder)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # ===========================================
    # 1️⃣ Adapt patch embedding to 2-channel input
    # ===========================================
    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None)
    )

    with torch.no_grad():
        # copy first 2 channels from original pretrained conv
        new_conv.weight[:, :2] = old_conv.weight[:, :2]

    model.segformer.encoder.patch_embeddings[0].proj = new_conv

    # ===========================================
    # 2️⃣ Reinitialize decoder from scratch
    # ===========================================
    def init_weights_kaiming(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.decode_head.apply(init_weights_kaiming)
    
    return model
