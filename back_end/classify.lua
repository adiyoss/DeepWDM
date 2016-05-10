----------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'rnn'
--require('mobdebug').start()
----------------------------------------------------------------------

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-folder_path', '/Users/yossiadi/Projects/wdm_lstm/torch/front_end/data/test_features/', 'the path to the data files')
cmd:option('-x_filename', 'x_test_naming.txt', 'the path to the featues')
cmd:option('-input_dim', 39, 'the input size')
cmd:option('-class_path', 'measurement/cls.txt', 'the path for the output file, hard classification')
cmd:option('-prob_path', 'measurement/prb.txt', 'the path for the output file, probability')

-- model:
cmd:option('-model_path', 'results/model.net', 'the path to the model')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> Measuring word duration'
dofile 'utils.lua'

print '==> loading data'
data_x = load_data(paths.concat(opt.folder_path, opt.x_filename), opt.input_dim)
outputs = torch.zeros(data_x:size(1))
prbs = torch.zeros(data_x:size(1))
softmax = nn.SoftMax()

print '==> loading model'
model = torch.load(opt.model_path)

x ={}
table.insert(x, data_x)
local output = model:forward(x)
output = softmax:forward(output[1])

for t=1,data_x:size(1) do
  outputs[t] = argmax(output[t])
  prbs[t] = output[t][2]
end

-- save the predictions
save_tensor_2_txt(outputs, opt.class_path)
save_tensor_2_txt(prbs, opt.prob_path)
