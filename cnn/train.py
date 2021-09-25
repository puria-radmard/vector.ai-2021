from tqdm import tqdm
from cnn.utils import *
import time


def epoch_process(model, criterion, optimizer, dataloader, mode):

    assert mode in ["train", "eval"]

    epoch_loss = 0
    total_correct = 0
    counter = 0

    for i, data in tqdm(enumerate(dataloader)):
        if mode == "train":
            optimizer.zero_grad()

        inputs, labels = data["X"], data["y"]
        batch_size = inputs.shape[0]

        output = model(inputs)
        loss = criterion(output, labels)
        if mode == "train":
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        total_correct += top_1_accuracy(output, labels)
        counter += batch_size

    return epoch_loss, total_correct, counter


def train(
    model,
    criterion,
    optimizer,
    train_dataloader,
    eval_dataloader,
    num_epochs,
    reloader=None,
):

    train_losses, eval_losses, train_accuracy, eval_accuracy = [], [], [], []

    for epoch in range(num_epochs):

        model.train()
        train_epoch_loss, train_total_correct, train_counter = epoch_process(
            model, criterion, optimizer, train_dataloader, "train"
        )
        print(
            f"Epoch {epoch}: train loss {train_epoch_loss/train_counter}, train accuracy {train_total_correct/train_counter}"
        )
        train_losses.append(train_epoch_loss / train_counter)
        train_accuracy.append(train_total_correct / train_counter)

        time.sleep(2)

        model.eval()
        eval_epoch_loss, eval_total_correct, eval_counter = epoch_process(
            model, criterion, None, eval_dataloader, "eval"
        )
        print(
            f"Epoch {epoch}: test loss {eval_epoch_loss/eval_counter}, test accuracy {eval_total_correct/eval_counter}"
        )
        eval_losses.append(eval_epoch_loss / eval_counter)
        eval_accuracy.append(eval_total_correct / eval_counter)

    return model
