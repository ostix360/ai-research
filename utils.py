def debug_data_processing(train_dataloader):
    batch = None
    for batch in train_dataloader:
        break
    print({k: v.shape for k, v in batch.items()})
    return batch


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

