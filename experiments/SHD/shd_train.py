import torch.nn
import tools
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import math
import random

from brf_snn.models import SimpleResRNN


from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, ExponentialLR, LinearLR

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(device)

################################################################
# Dataset
################################################################

# TRAIN DATASET #
whole_train_dataset = tools.shd_to_dataset('./data/trainX_4ms.npy', './data/trainY_4ms.npy')

# 8156 sequences in whole training dataset
total_train_dataset_size = len(whole_train_dataset)

# 10 % of training data used for validation -> 815
val_dataset_size = int(total_train_dataset_size * 0.1)

# 7341 sequences used for training
train_dataset_size = total_train_dataset_size - val_dataset_size

# split whole train dataset randomly
train_dataset, val_dataset = random_split(
    dataset=whole_train_dataset,
    lengths=[train_dataset_size, val_dataset_size]
)

# TEST DATASET #
test_dataset = tools.shd_to_dataset('./data/testX_4ms.npy', './data/testY_4ms.npy')

# 2264 sequences in test dataset
test_dataset_size = len(test_dataset)

####################################################################
# DataLoader
####################################################################

sequence_length = 250
input_size = 700
hidden_size = 128

num_classes = 20
batch_size = 32

# validation and test batch size can be chosen higher
# (depending on VRAM capacity)
val_batch_size = 256
test_batch_size = 256

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False
)

################################################################
# Model helpers and model setup
################################################################

delta = 0.01 #4e-3

# recorded into comment
# fraction of the elements in the hidden.linear.weight to be zero
mask_prob = 0.0

# omega init uniform distribution CHECK on rf.py!
omega_a = 5.
omega_b = 10.

# b_offset init uniform distribution CHECK on rf.py!
b_offset_a = 2.
b_offset_b = 3.

# LI tau_mem init normal distribution
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 5.

label_last = False
sub_seq_length = 0

hidden_bias = False
output_bias = False

model = SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_a,
    adaptive_omega_b=omega_b,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    label_last=label_last,
    sub_seq_length=sub_seq_length,
    mask_prob=mask_prob,
    output_bias=False,
    dt=delta
).to(device)

# TORCH SCRIPT #
model = torch.jit.script(model)

# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(num_params)
# 108820

################################################################
# Setup experiment (optimizer etc.)
################################################################

# prevent overwriting in slurm
rand_num = random.randint(1, 10000)

criterion = torch.nn.NLLLoss()  # CrossEntropyLoss()

optimizer_lr = 0.075
gradient_clip_value = 1.


optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

# Number of iterations per epoch
total_steps = len(train_loader)
epochs_num = 20

# learning rate scheduling
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs_num)

# [logging] Only thing manually changed in the string: Optimizer, criterion and scheduler!
opt_str = "{}_Adam({}),NLL,script-fgiDG,LinLR,LL({},no_gc)".format(rand_num, optimizer_lr, label_last)
net_str = "700,{},20,bs_{},ep_{}"\
    .format(hidden_size, batch_size, epochs_num)
unit_str = "BRF(omega{},{}b{},{})LI({},{})"\
    .format(omega_a, omega_b, b_offset_a, b_offset_b, out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)

comment = opt_str + "," + net_str + "," + unit_str

writer = SummaryWriter(comment=comment)
start_time = datetime.now().strftime("%m-%d_%H-%M-%S")


save_path = "models/{}_".format(start_time) + comment + ".pt"
save_init_path = "models/{}_init_".format(start_time) + comment + ".pt"

# save initial parameters for analysis
torch.save({'model_state_dict': model.state_dict()}, save_init_path)

# print(model.state_dict())
print_every = 230
print(comment)

################################################################
# Training loop
################################################################

iteration = 0
min_val_loss = float("inf")

# init dummy loss_value at beginning
loss_value = 1.
end_training = False

run_time = tools.PerformanceCounter()
tools.PerformanceCounter.reset(run_time)

for epoch in range(epochs_num + 1):

    # check initial performance without training (for plotting purposes)
    # Go into eval mode
    model.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0

        # Perform validation
        for i, (inputs, targets) in enumerate(val_loader):

            # Reshape inputs in [sequence_length, batch_size, data_size].
            input = inputs.permute(1, 0, 2).to(device)

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            outputs, _, _ = model(input)

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                val_loss_value = loss.item()
            else:
                val_loss_value = loss.item() / (sequence_length - sub_seq_length)

            val_loss += val_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
            val_correct += batch_correct

        val_loss /= len(val_loader)  # val_dataset_size
        val_accuracy = (val_correct / val_dataset_size) * 100.0

        # Log current val loss and accuracy
        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/val",
            val_accuracy,
            epoch
        )

        # Persist current best model.
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_model_state_dict = model.state_dict()
            # TODO save checkpoint of the training including model.state_dict() and optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, save_path)

        test_loss = 0
        test_correct = 0
        test_total_spikes = 0

        # Perform Inference
        for i, (inputs, targets) in enumerate(test_loader):

            # Reshape inputs in [sequence_length, batch_size, data_size].
            input = inputs.permute(1, 0, 2).to(device)

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            outputs, _, num_spikes1 = model(input)

            # accumulate total spikes
            test_total_spikes += num_spikes1.item()

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                test_loss_value = loss.item()
            else:
                test_loss_value = loss.item() / (sequence_length - sub_seq_length)

            test_loss += test_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
            test_correct += batch_correct

        test_loss /= len(test_loader)  # test_dataset_size
        test_accuracy = (test_correct / test_dataset_size) * 100.0
        test_sop = test_total_spikes / test_dataset_size

        # Log current test loss and accuracy
        writer.add_scalar(
            "Loss/test",
            test_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/test",
            test_accuracy,
            epoch
        )

        print(
            "Epoch [{:4d}/{:4d}]  |  Summary  |  Loss/val: {:.6f}, Accuracy/val: {:.4f}%  |  Loss/test: {:.6f}, "
            "Accuracy/test: {:.4f} | SOP: {}".format(
                epoch, epochs_num, val_loss, val_accuracy, test_loss, test_accuracy, test_sop), flush=True
        )

    # Update logging outputs
    writer.flush()

    # TRAIN #
    # Run until 19 epochs, the last loop only for the test/valid.
    if epoch < epochs_num:

        # Go into train mode.
        model.train()

        print_train_loss = 0
        print_correct = 0
        print_total = 0

        # Perform training epoch (iterate over all mini batches in training set).
        for i, (inputs, targets) in enumerate(train_loader):

            current_batch_size = len(inputs)

            # Reshape inputs in [sequence_length, batch_size, data_size].
            input = inputs.permute(1, 0, 2).to(device)

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            # Clear previous gradients
            optimizer.zero_grad()

            outputs, _, _ = model(input)

            # Apply loss sequentially against single pattern.
            loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                loss_value = loss.item()
            else:
                loss_value = loss.item() / (sequence_length - sub_seq_length)

            # calculate gradient
            loss.backward()

            if math.isnan(loss_value):
                end_training = True
                break

            # Perform learning step
            optimizer.step()

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)

            # Log current loss and accuracy
            writer.add_scalar(
                "Loss/train",
                loss_value,
                iteration
            )
            writer.add_scalar(
                "Accuracy/train",
                (batch_correct / current_batch_size) * 100.0,
                iteration
            )

            print_train_loss += loss_value
            print_total += current_batch_size
            print_correct += batch_correct

            # Print current training loss/acc at every 50th iteration
            if i % print_every == (print_every - 1):
                print_acc = (print_correct / print_total) * 100.0

                print("Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f}, Accuracy/train: {:8.4f}".format(
                    epoch + 1, epochs_num, i + 1, total_steps, print_train_loss / print_every, print_acc), flush=True
                )

                print_correct = 0
                print_total = 0
                print_train_loss = 0

            iteration += 1

        scheduler.step()

        # Update logging outputs
        writer.flush()

    if end_training:
        break

writer.close()
print(tools.PerformanceCounter.time(run_time), "seconds")
print("Minimum val loss: {:.6f} at epoch: {}".format(min_val_loss, min_val_epoch))
# print(best_model_state_dict)

