----------------------------------------------------------------------
require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'rnn'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Building the model')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'rnn', 'type of model to construct: mlp | convnet | rnn')
   cmd:option('-input_dim', 39, 'the input size')
   cmd:option('-drop_out', 0.5, 'dropout rate')
   cmd:option('-output_dim', 2, 'the output size')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 2-class problem
noutputs = opt.output_dim

-- number of hidden units (for MLP only):
nhiddens_1 = opt.input_dim * 2
nhiddens_2 = opt.input_dim

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'mlp' then
   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Linear(opt.input_dim, nhiddens_1))
   model:add(nn.Dropout(opt.dropout))
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens_1, nhiddens_2))
   model:add(nn.Dropout(opt.dropout)) 
   model:add(nn.ReLU())
   model:add(nn.Linear(nhiddens_2, noutputs))
   
elseif opt.model == 'convnet' then  
  error('not supported yet')
  
elseif opt.model == 'birnn' then    
  -- forward rnn
  -- build LSTM based rnn
  fwd = nn.GRU(opt.input_dim, nhiddens_1)

  -- backward rnn (will be applied in reverse order of input sequence)
  bwd = fwd:clone()
  bwd:reset() -- reinitializes parameters

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  merge = nn.JoinTable(1, 1)
  
  -- build the bidirectional lstm
  brnn = nn.BiSequencer(fwd, bwd, merge)

  model = nn.Sequential()
     :add(brnn) 
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(2*nhiddens_1, opt.output_dim))) -- times two due to JoinTable
     :add(nn.Sequencer(nn.LogSoftMax()))

elseif opt.model == 'rnn' then
  --- ONE LAYER RNN ---
  fwd_1 = nn.GRU(opt.input_dim, nhiddens_1)
  s_rnn_1 = nn.Sequencer(fwd_1)

  model = nn.Sequential()     
     :add(s_rnn_1)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(nhiddens_1, opt.output_dim)))
     :add(nn.Sequencer(nn.LogSoftMax()))

elseif opt.model == '2rnn' then
  --- TWO LAYERS RNN ---
  fwd_1 = nn.GRU(opt.input_dim, nhiddens_1)
  fwd_2 = nn.GRU(nhiddens_1, nhiddens_2)
  s_rnn_1 = nn.Sequencer(fwd_1)
  s_rnn_2 = nn.Sequencer(fwd_2)

  model = nn.Sequential()     
     :add(s_rnn_1)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(s_rnn_2)
     :add(nn.Sequencer(nn.Dropout(opt.dropout)))
     :add(nn.Sequencer(nn.Linear(nhiddens_2, opt.output_dim)))
     :add(nn.Sequencer(nn.LogSoftMax()))
else
  error('unknown -model')
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
