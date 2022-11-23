import torch
import time
import numpy as np
from dice_coef import dice_coef, iou_pytorch
from model import model_act
from dataset import train_val_dataloader
from preproc import preproc
import matplotlib.pyplot as plt


def train(seg_model, opt, scheduler, loss_fn, epochs, data_tr, data_val, score=True):
    device = 'cuda'
    history_train_loss = []
    history_val_loss = []
    best_loss = np.inf
    best_metric = -1
    if score:
        history_metric = []
        history_dice = []

    start_time = time.time()
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch + 1, epochs))

        train_avg_loss, val_avg_loss = 0, 0
        seg_model.train()
        for data in data_tr:
            X_batch = data['image'].cuda()
            Y_batch = data['mask'].cuda()

            opt.zero_grad()

            output = seg_model(X_batch)

            loss = loss_fn(output, Y_batch.long())
            loss.backward()
            opt.step()

            train_avg_loss += loss.item() / len(data_tr)

        scheduler.step()
        history_train_loss.append(train_avg_loss)

        seg_model.eval()
        scores = 0
        dice = 0
        with torch.no_grad():
            for data in data_val:
                X_batch = data['image'].to(device)
                Y_batch = data['mask'].to(device)

                output = seg_model(X_batch)
                Y_pred = output

                scores += iou_pytorch(Y_pred.argmax(1).long(), Y_batch.long()).mean().item()
                dice += dice_coef(Y_pred.argmax(1).long(), Y_batch.long()).mean().item()

                loss = loss_fn(output, Y_batch.long())

                val_avg_loss += loss.item() / len(data_val)

            history_val_loss.append(val_avg_loss)
            metric = scores / len(data_val)
            dice_p = dice / len(data_val)

            elapsed_time = time.time() - start_time
            loss_log = f'Train loss: {train_avg_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            current_model_log = f'{"Current_metric":17s}: {metric:0.3f}, {"Current_loss":17s}: {val_avg_loss:0.2f}'

            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                torch.save(seg_model.state_dict(), f'/output/{epoch + 1}_epoch_best_loss{best_loss:.4f}.pth')

            if metric > best_metric:
                best_metric = metric
                torch.save(seg_model.state_dict(), f'/output/{epoch + 1}_epoch_best_metric{best_metric:.4f}.pth')

            if score:
                history_metric.append(metric)
                history_dice.append(dice_p)

        best_model_log = f'{"Best_metric":17s}: {best_metric:0.3f}, {"Best_loss":17s}: {best_loss:0.2f}'
        loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'

        print(loss_model_log)
        dashed_line = '-' * 80
        print(dashed_line)
    if score:
        return history_train_loss, history_val_loss, metric, history_dice
    else:
        return history_train_loss, history_val_loss


def main():
    df = preproc()
    train_dataloader, val_dataloader = train_val_dataloader(df)
    seg_model, device, optimizer, loss_fn, scheduler = model_act()
    train_loss, val_loss, score, dice = train(seg_model, optimizer, scheduler, loss_fn, 20,
                                              train_dataloader, val_dataloader)

    plt.plot(range(1, 21), dice)
    plt.xticks(range(1, 21))
    plt.xlabel('Epoch')
    plt.ylabel('Dice_coef')
    plt.savefig('/subset/output/output.png')