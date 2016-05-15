----------------------------------------------------------------------
require 'torch'
--require('mobdebug').start()

----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 4, 'number of threads')
cmd:option('-patience', 5, 'the number of epochs to be patient before early stopping')
cmd:option('-epsilon', 0.01, 'the minimum amount of loss improvment require for keep training the model')
-- data:
cmd:option('-folder_path', 'data/', 'the path to the data files')
cmd:option('-x_filename', 'x.t7', 'the features filename')
cmd:option('-y_filename', 'y.t7', 'the labels filename')
cmd:option('-input_dim', 39, 'the input size')
cmd:option('-output_dim', 2, 'the output size')
cmd:option('-val_percentage', 0.2, 'the percentage of exampels to be considered as validation set from the training set')
-- model:
cmd:option('-model', 'birnn', 'type of model to construct: mlp | convnet | rnn | birnn')
cmd:option('-drop_out', 0.5, 'dropout rate')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'ADAGRAD', 'optimization method: SGD | ADAM | ADAGRAD')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 64, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-rho', 100, 'max sequence length')
cmd:text()
opt = cmd:parse(arg or {})

paramsLogger = io.open(paths.concat(opt.save, 'params.log'), 'w')
-- save cmd parameters
for key, value in pairs(opt) do
  paramsLogger:write(key .. ': ' .. tostring(value) .. '\n')
end
paramsLogger:close()

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
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'validate.lua'

----------------------------------------------------------------------
print '==> training!'
local iteration = 1
local best_loss = 100000
local loss = 0

loss = validate()
print('==> validation loss: ' .. loss)

-- training
while loss + opt.epsilon < best_loss or iteration < opt.patience do       
  
  -- train - forward and backprop
  train() 
    
  -- validate  
  loss = validate()
  print('\n==> validation loss: ' .. loss)
  -- for early stopping criteria
  if loss + opt.epsilon >= best_loss then 
    -- increase iteration number
    iteration = iteration + 1
    print('\n========================================')
    print('==> Loss did not improved, iteration: ' .. iteration)
    print('========================================\n')
  else
    -- update the best loss value
    best_loss = loss
    
    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)    
    torch.save(filename, model)
    iteration = 1
  end
end

--[[
-- evaluate on the test set
local test_loss = test(test_x, test_y)
print('\n============ EVALUATING ON TEST SET ============')
print('Loss = ' .. test_loss .. '\n')
]]--
