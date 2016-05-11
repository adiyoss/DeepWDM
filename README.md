# DeepWDM - Recurrent neural networks for Word Duration Measurement written in [Torch7](http://torch.ch)

## Content
The repository contains code for word duration measurement.
 - `back_end folder`: contains the training algorithms, it can be used for training the model on new datasets or using different features.
 - `front_end folder`: contains the features extraction algorithm.
 - `lib folder`: contains some useful python scripts.
 - `data folder`: contains the example file to test the repository.

## Installation
The code runs on MacOSX only.
### Dependencies
The code uses the following dependencies:
 - Torch7 with RNN package
```bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh 

# On Linux with bash
source ~/.bashrc
# On Linux with zsh
source ~/.zshrc
# On OSX or in Linux with none of the above.
source ~/.profile

# For rnn package installation
luarocks install rnn
```
 - [Python (2.7) + Numpy] (https://penandpants.com/2012/02/24/install-python/)
 - For the visualization tools: [Matplotlib] (https://penandpants.com/2012/02/24/install-python/)
 
### Model Installation
Download the model from: [DeepWDM Model] (??). Than, move the model file to: `back_end/results/` inside the project directory.

## Usage
For measurement just type: 
```bash
python predict.py "input wav file" "output text grid file"
```

## Example
You can try our tool using the example file in the data folder. 
Type:
```bash
python predict.py data/test.wav data/test.TextGrid
```
