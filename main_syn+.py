from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils
from model import Net
from data import get_training_set, get_test_set, get_syn_training_set, get_syn1_test_set, get_syn2_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--trainCG', action='store_true', help='use CG data for training?')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
if opt.trainCG:
    #CG - level_Design train dataset  (약 25,000)
    train_set = get_syn_training_set(opt.upscale_factor)
else:
    #Real Image - ImageNet  (50,000)
    train_set = get_training_set(opt.upscale_factor, True)  # ImageNet 사용 = True
#BSDS300 test 사용
test_set = get_test_set(opt.upscale_factor)
#(CG)dataset/valid/test 사용 
syn1_test_set = get_syn1_test_set(opt.upscale_factor)
#(CG)dataset/CG/test 사용 
syn2_test_set = get_syn2_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
syn1_testing_data_loader = DataLoader(dataset=syn1_test_set, num_workers=opt.threads, batch_size=72, shuffle=False)
syn2_testing_data_loader = DataLoader(dataset=syn2_test_set, num_workers=opt.threads, batch_size=28, shuffle=False)

print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor).to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(): # for real images
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    
def test_syn1():  # for synthetic(virtual) images
    avg_psnr = 0
    n = 0
    with torch.no_grad():
        for batch in syn1_testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            print(prediction.size)
            #for i in range(prediction.size(0)):
            #    torchvision.utils.save_image(prediction[i, :, :, :], 'batch{}_{}.png'.format(n,i))
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
            n += 1
    print("===> Avg. PSNR (VALID): {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def test_syn2():  # for synthetic(virtual) images
    avg_psnr = 0
    with torch.no_grad():
        for batch in syn2_testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR (CG): {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    
def checkpoint(epoch):
    if opt.trainCG:
        model_out_path = "CG_model_epoch_{}.pth".format(epoch)
    else:
        model_out_path = "Real_model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def main():
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
        test_syn1()
        test_syn2()
        checkpoint(epoch)

if __name__ == '__main__':
    main()
