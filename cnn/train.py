from utils import train_epoch, val_epoch


def train(model, criterion, optimizer, dataloader, num_epochs, reloader=None):

    train_losses, eval_losses, train_accuracy, eval_accuracy = [], [], [], []

    for epoch in range(num_epochs):

        model.train()
        train_epoch_loss, train_total_correct, train_counter = epoch_process(
            model, criterion, optimizer, dataloader, "train"
        )
        print(
            f"Epoch {epoch}: train loss {train_epoch_loss/train_counter}, train accuracy {train_total_correct/train_counter}"
        )
        train_losses.append(train_epoch_loss / train_counter)
        train_accuracy.append(train_total_correct / train_counter)

        model.eval()
        eval_epoch_loss, eval_total_correct, eval_counter = epoch_process(
            model, criterion, dataloader, "eval"
        )
        print(
            f"Epoch {epoch}: test loss {eval_epoch_loss/eval_counter}, test accuracy {eval_total_correct/eval_counter}"
        )
        eval_losses.append(eval_epoch_loss / eval_counter)
        eval_accuracy.append(eval_total_correct / eval_counter)

    return model
