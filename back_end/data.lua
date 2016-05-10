--require('mobdebug').start()
require ('torch')
require ('nn')
require ('rnn')
dofile ('utils.lua')

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print('==> processing options')
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Loading data')
   cmd:text()
   cmd:text('Options:')
   cmd:option('folder_path', '/Users/yossiadi/Projects/wdm_lstm/torch/front_end/data/measurement/', 'the path to the data files')
   cmd:option('-input_dim', 39, 'the input size')
   cmd:option('-output_dim', 2, 'the output size')
   cmd:option('-val_percentage', 0.2, 'the percentage of exampels to be considered as validation set from the training set')
   cmd:text()
   opt = cmd:parse(args or {})
end
----------------------------------------------------------------------

mini_batch_idx = 0 --for the minibatchs
local output_dim = opt.output_dim

print("==> Loading data set")

data_dir = opt.folder_path

--[[
local x_filename = 'x_train_naming.txt'
local y_filename = 'y_train_naming.txt'
local data_x = load_data(paths.concat(data_dir, x_filename), opt.input_dim)
local tmp_y = load_data(paths.concat(data_dir, y_filename), output_dim)
local data_y = tmp_y[{{}, 1}]


torch.save('x.t7', data_x)
torch.save('y.t7', data_y) 
]]--

local data_x = torch.load('data/x.t7')
local data_y = torch.load('data/y.t7')

-- Detecting and removing NaNs
if data_x:ne(data_x):sum() > 0 then
  print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
  data_x[data_x:ne(data_x)] = 0
end
if data_y:ne(data_y):sum() > 0 then
  print(sys.COLORS.red .. ' training set has NaN/s, replace with zeros.')
  data_y[data_y:ne(data_y)] = 0
end

-- take part of the training set for validation
local val_size = data_x:size(1)*opt.val_percentage

val_x = data_x[{{(data_x:size(1)-val_size), data_x:size(1)}, {}}]   -- take the last elements for validation
val_y = data_y[{{(data_y:size(1)-val_size), data_y:size(1)}}]   -- take the last elements for validation

local max_size = 400000
--train_x = data_x[{{1, (data_x:size(1)-val_size)}, {}}]            -- the rest are for training
--train_y = data_y[{{1, (data_y:size(1)-val_size)}}]            -- the rest are for training

train_x = data_x[{{1, max_size}, {}}]            -- the rest are for training
train_y = data_y[{{1, max_size}}]            -- the rest are for training

local p_train = torch.sum(train_y) / train_y:size(1)
local p_val = torch.sum(val_y) / val_y:size(1)

train_y:add(1)
val_y:add(1)

trainData = {
   data = train_x,
   labels = train_y,
   size = function() return train_x:size(1) end
}

valData = {
   data = val_x,
   labels = val_y,
   size = function() return val_x:size(1) end
}

print '==> data statistics:'
print('==> training set size: ' .. trainData:size() .. ', label 1: ' .. (1 - p_train) .. ', label 2:' .. p_train)
print('==> validation set size: ' .. valData:size() .. ', label 1: ' .. (1 - p_val) .. ', label 2:' .. p_val)

