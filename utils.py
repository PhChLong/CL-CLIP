def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total:     {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {frozen:,}")