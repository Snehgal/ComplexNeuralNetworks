# ===============================
# 2️⃣ Model Setup
# ===============================
def build_segformer(num_classes=9, pretrained_encoder="nvidia/segformer-b5-finetuned-ade-640-640"):
    # Load pretrained model
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_encoder,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # allows decoder resizing
    )

    # -------------------------
    # Re-initialize decoder and classifier heads with Kaiming
    # -------------------------
    def kaiming_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Segformer decoder layers
    model.decode_head.apply(kaiming_init)
    

    # Adapt first conv layer for 2-channel input
    old_conv = model.segformer.encoder.patch_embeddings[0].proj
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    # Average pretrained weights for 2 channels
    with torch.no_grad():
        new_conv.weight[:, :2] = old_conv.weight[:, :2]
    model.segformer.encoder.patch_embeddings[0].proj = new_conv

    return model
