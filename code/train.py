
import torch

def train(model, train_data, optimizer, opt):
    model.train()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score,x,y = model(train_data)
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = loss(score, train_data['M_D'].cpu())
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch + 1, "Loss:", loss.item())
    return model




