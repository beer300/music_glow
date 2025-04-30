# MIT License
# 
# Copyright (c) 2025 Ke Fang
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable

import time
import math
import argparse
cuda = True if torch.cuda.is_available() else False
import librosa
import numpy as np
import torch
import soundfile

from packaging import version

N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32


class RandomCNN(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()

        # 2-D CNN
        self.conv1 = nn.Conv2d(1, OUT_CHANNELS, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS * 2, kernel_size=(3, 3), stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(OUT_CHANNELS * 2)
        self.dropout = nn.Dropout(0.5)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.LeakyReLU(self.conv1(x))
        x = self.batch_norm(self.conv2(x))
        x = self.dropout(x)
        return x
    
class RandomCNN_V1(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()

        # 2-D CNN
        self.conv1 = nn.Conv2d(1, OUT_CHANNELS, kernel_size=(3, 1), stride=1, padding=0)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the random parameters to be constant.
        weight = torch.randn(self.conv1.weight.data.shape)
        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias = torch.zeros(self.conv1.bias.data.shape)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_delta):
        out = self.LeakyReLU(self.conv1(x_delta))
        return out

"""
a_random = Variable(torch.randn(1, 1, 257, 430)).float()
model = RandomCNN()
a_O = model(a_random)
print(a_O.shape)
"""
def librosa_write(outfile, x, sr):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(outfile, x, sr)
    else:
        soundfile.write(outfile, x, sr)

def wav2spectrum(filename):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, sr


def spectrum2wav(spectrum, sr, outfile):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=N_FFT))
    librosa_write(outfile, x, sr)


def wav2spectrum_keep_phase(filename):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft=N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, p, sr


def spectrum2wav_keep_phase(spectrum, p, sr, outfile):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft=N_FFT))
    librosa_write(outfile, x, sr)


def compute_content_loss(a_C, a_G):
    """
    Compute the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

    # Compute the cost
    J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

    return J_content


def gram(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_L)

    Returns:
    GA -- Gram matrix of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)

    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C, n_H, n_W = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA


def compute_layer_style_loss(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)

    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

    return J_style_layer


OUTPUT_DIR = r"C:\Users\lukas\Documents\new_model_outputs\out_2"



parser = argparse.ArgumentParser()
parser.add_argument('-content', help='Content input')
parser.add_argument('-content_weight', help='Content weight. Default is 1e2', default = 1e2)
parser.add_argument('-style', help='Style input')
parser.add_argument('-style_weight', help='Style weight. Default is 1', default = 1)
parser.add_argument('-epochs', type=int, help='Number of epoch iterations. Default is 20000', default = 5000)
parser.add_argument('-print_interval', type=int, help='Number of epoch iterations between printing losses', default = 50)
parser.add_argument('-plot_interval', type=int, help='Number of epoch iterations between plot points', default = 1000)
parser.add_argument('-learning_rate', type=float, default = 0.002)

args = parser.parse_args()


CONTENT_FILENAME = args.content
STYLE_FILENAME = args.style

a_content, sr = wav2spectrum(CONTENT_FILENAME)
a_style, sr = wav2spectrum(STYLE_FILENAME)

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(f"content", a_content_torch.shape)
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
if cuda:
    a_style_torch = a_style_torch.cuda()
print(a_style_torch.shape)

model = RandomCNN()
model.eval()

a_C_var = Variable(a_content_torch, requires_grad=False).float()
a_S_var = Variable(a_style_torch, requires_grad=False).float()
if cuda:
    model = model.cuda()
    a_C_var = a_C_var.cuda()
    a_S_var = a_S_var.cuda()

a_C = model(a_C_var)
a_S = model(a_S_var)


# Optimizer
learning_rate = args.learning_rate
a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3)
if cuda:
    a_G_var = a_G_var.cuda()
a_G_var.requires_grad = True
optimizer = torch.optim.Adam([a_G_var])

# coefficient of content and style
style_param = args.style_weight
content_param = args.content_weight

num_epochs = args.epochs
print_every = args.print_interval
plot_every = args.plot_interval

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class EarlyStopping:
    """
    Early stopping to stop training when the validation loss does not improve after a given patience.
    """
    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(model, a_C, a_S, a_G_var, optimizer, content_weight, style_weight, num_epochs, print_every, plot_every, patience=10):
    """
    Train the model with early stopping to optimize the generated audio spectrum (a_G_var).
    
    Args:
        model (nn.Module): Pre-trained RandomCNN model used to extract features.
        a_C (torch.Tensor): Content features extracted from the content audio.
        a_S (torch.Tensor): Style features extracted from the style audio.
        a_G_var (torch.Tensor): Initial generated spectrum to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer for updating a_G_var.
        content_weight (float): Weight for the content loss.
        style_weight (float): Weight for the style loss.
        num_epochs (int): Number of training epochs.
        print_every (int): Interval for printing training progress.
        plot_every (int): Interval for recording losses for plotting.
        patience (int): Number of epochs to wait for improvement before stopping.
    
    Returns:
        list: List of average losses recorded at each plot interval.
    """
    model.eval()  # Ensure the model is in evaluation mode
    current_loss = 0.0
    all_losses = []
    start = time.time()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Forward pass to get the generated features
        a_G = model(a_G_var)

        # Compute style loss every iteration
        style_loss = style_weight * compute_layer_style_loss(a_S, a_G)
        style_loss.backward(retain_graph=True)

        # Compute content loss every second iteration
        if epoch % 2 == 0:
            content_loss = content_weight * compute_content_loss(a_C, a_G)
            content_loss.backward(retain_graph=True)
        else:
            content_loss = torch.tensor(0.0)  # No content loss update on odd iterations
            if cuda:
                content_loss = content_loss.cuda()

        # Perform optimization step
        optimizer.step()

        # Compute total loss for logging
        total_loss = content_loss + style_loss

        # Print training progress
        if epoch % print_every == 0:
            print(f"{epoch} {epoch/num_epochs*100:.2f}% {timeSince(start)} "
                  f"content_loss: {content_loss.item():.4f} style_loss: {style_loss.item():.4f} total_loss: {total_loss.item():.4f}")
            current_loss += total_loss.item()

        # Save the generated audio every 500 epochs
        if epoch % 500 == 0:
            gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
            gen_audio_filename = f"{OUTPUT_DIR}_epoch_{epoch}.wav"
            spectrum2wav(gen_spectrum, sr, gen_audio_filename)
            print(f"Saved audio at epoch {epoch}: {gen_audio_filename}")

        # Record losses for plotting
        if epoch % plot_every == 0:
            avg_loss = current_loss / plot_every
            all_losses.append(avg_loss)
            current_loss = 0.0

            # Check early stopping
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return all_losses

# Train the model using the refactored train function
all_losses = train(
    model=model,
    a_C=a_C,
    a_S=a_S,
    a_G_var=a_G_var,
    optimizer=optimizer,
    content_weight=content_param,
    style_weight=style_param,
    num_epochs=num_epochs,
    print_every=print_every,
    plot_every=plot_every,
    patience=10  # Stop training if no improvement for 10 epochs
)

# Generate the final spectrum and save outputs
gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
gen_audio_C = OUTPUT_DIR + ".wav"
spectrum2wav(gen_spectrum, sr, gen_audio_C)

plt.figure()
plt.plot(all_losses)
plt.savefig('loss_curve.png')

plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
plt.title("Content Spectrum")
plt.imsave('Content_Spectrum.png', a_content[:400, :])

plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
plt.title("Style Spectrum")
plt.imsave('Style_Spectrum.png', a_style[:400, :])

plt.figure(figsize=(5, 5))
plt.subplot(1, 1, 1)
plt.title("CNN Voice Transfer Result")
plt.imsave('Gen_Spectrum.png', gen_spectrum[:400, :])






"""
# Test
test_S = torch.randn(1, 6, 2, 2)
test_G = torch.randn(1, 6, 2, 2)
print(test_S)
print(test_G)
print(compute_layer_style_loss(test_S, test_G))


# Test
test_C = torch.randn(1, 6, 2, 2)
test_G = torch.randn(1, 6, 2, 2)
print(test_C)
print(test_G)
print(compute_content_loss(test_C, test_G))
"""