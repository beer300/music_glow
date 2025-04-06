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
parser.add_argument('-content_weight', help='Content weight. Default is 1e2', default=1e2)
parser.add_argument('-epochs', type=int, help='Number of epoch iterations. Default is 2000', default=2000)
parser.add_argument('-print_interval', type=int, help='Number of epoch iterations between printing losses', default=1000)
parser.add_argument('-plot_interval', type=int, help='Number of epoch iterations between plot points', default=1000)
parser.add_argument('-learning_rate', type=float, default=0.002)

args = parser.parse_args()

CONTENT_FILENAME = args.content

# Load only the content input
a_content, sr = wav2spectrum(CONTENT_FILENAME)

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(a_content_torch.shape)

model = RandomCNN()
model.eval()

a_C_var = Variable(a_content_torch, requires_grad=False).float()
if cuda:
    model = model.cuda()
    a_C_var = a_C_var.cuda()

a_C = model(a_C_var)

# Optimizer
learning_rate = args.learning_rate
a_G_var = Variable(torch.randn(a_content_torch.shape) * 1e-3)
if cuda:
    a_G_var = a_G_var.cuda()
a_G_var.requires_grad = True
optimizer = torch.optim.Adam([a_G_var])

# Coefficient of content loss
content_param = args.content_weight

num_epochs = args.epochs
print_every = args.print_interval
plot_every = args.plot_interval

def timeSince(start):
    """
    Calculate the elapsed time since the start time and format it as a string.

    Args:
        start (float): Start time (from time.time()).

    Returns:
        str: Formatted string showing elapsed time in minutes and seconds.
    """
    now = time.time()
    elapsed = now - start
    minutes = math.floor(elapsed / 60)
    seconds = elapsed - minutes * 60
    return f"{minutes}m {seconds:.2f}s"

# Modify the train function to optimize only for content loss
def train_content_only(model, a_C, a_G_var, optimizer, content_weight, num_epochs, print_every, plot_every):
    """
    Train the model to optimize the generated audio spectrum (a_G_var) to minimize content loss only.
    
    Args:
        model (nn.Module): Pre-trained RandomCNN model used to extract features.
        a_C (torch.Tensor): Content features extracted from the content audio.
        a_G_var (torch.Tensor): Initial generated spectrum to be optimized.
        optimizer (torch.optim.Optimizer): Optimizer for updating a_G_var.
        content_weight (float): Weight for the content loss.
        num_epochs (int): Number of training epochs.
        print_every (int): Interval for printing training progress.
        plot_every (int): Interval for recording losses for plotting.
    
    Returns:
        list: List of average losses recorded at each plot interval.
    """
    model.eval()  # Ensure the model is in evaluation mode
    current_loss = 0.0
    all_losses = []
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # Forward pass to get the generated features
        a_G = model(a_G_var)

        # Compute content loss
        content_loss = content_weight * compute_content_loss(a_C, a_G)

        # Backward pass and optimization
        content_loss.backward()
        optimizer.step()

        # Print training progress
        if epoch % print_every == 0:
            print(f"{epoch} {epoch/num_epochs*100:.2f}% {timeSince(start)} "
                  f"content_loss: {content_loss.item():.4f}")
            current_loss += content_loss.item()

        # Record losses for plotting
        if epoch % plot_every == 0:
            avg_loss = current_loss / plot_every
            all_losses.append(avg_loss)
            current_loss = 0.0

    return all_losses

# Train the model using the modified train function
all_losses = train_content_only(
    model=model,
    a_C=a_C,
    a_G_var=a_G_var,
    optimizer=optimizer,
    content_weight=content_param,
    num_epochs=num_epochs,
    print_every=print_every,
    plot_every=plot_every
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
plt.title("Generated Spectrum")
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