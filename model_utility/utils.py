import torch


def save_checkpoint(MODEL_PATH, state, filename):
    print("=> Saving checkpoint")
    torch.save(state, MODEL_PATH+filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



