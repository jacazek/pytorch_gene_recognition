import torch
import torch.nn as nn
import torch.optim as optim
import os

from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
device = "cuda"
dtype=torch.float32
# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        # # Initialize cell state with zeros
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 10
hidden_size = 32
num_layers = 1
output_size = 1
seq_length = 100
batch_size = 10
num_epochs = 10
learning_rate = 0.01

# Generate synthetic data
x_train = torch.randn(batch_size, seq_length, input_size, device=device)
y_train = torch.randn(batch_size, output_size, device=device)

def batch_generator(number):
    for i in range(number):
        yield x_train

# Initialize the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device=device)
model = torch.compile(model)
# model = torch.compile(model, "max-autotune")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, fused=True)
scaler = torch.cuda.amp.GradScaler()

# Training loop
#
with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            schedule=torch.profiler.schedule(wait=1, warmup=30, active=10, repeat=1), record_shapes=True
                            ) as prof:
    with record_function("model_inference"):
        for epoch in range(num_epochs):
            with tqdm(batch_generator(100), ) as train:
                train.set_description(f"Epoch: {epoch}")
                for batch in train:
                    # optimizer.zero_grad(set_to_none=True)
                    # Forward pass
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        outputs = model(batch)
                    # loss = criterion(outputs, y_train)

                    # # Backward and optimize

                    # scaler.scale(loss).backward()
                    # loss.backward()
                    # scaler.step(optimizer)
                    # optimizer.step()
                    # scaler.update()
                    prof.step()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # prof.export_chrome_trace("trace-nvidia-other.json")
