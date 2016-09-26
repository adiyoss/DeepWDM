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
   cmd:option('-folder_path', 'data/', 'the path to the data files')
   cmd:option('-x_filename', 'x.t7', 'the features filename')
   cmd:option('-y_filename', 'y.t7', 'the labels filename')
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
local data_x = {}
local data_y = {}

if string.sub(opt.x_filename,-string.len('.t7'))=='.t7' then
  data_x = torch.load(paths.concat(opt.folder_path, opt.x_filename))
  data_y = torch.load(paths.concat(opt.folder_path, opt.y_filename))
else
  data_x = load_data(paths.concat(opt.folder_path, opt.x_filename), opt.input_dim)
  local tmp_y = load_data(paths.concat(opt.folder_path, opt.y_filename), output_dim)
  data_y = tmp_y[{{}, 1}]
  --torch.save('x.t7', data_x)
  --torch.save('y.t7', data_y)
end

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

-- local max_size = 400000
-- train_x = data_x[{{1, max_size}, {}}]            -- the rest are for training
-- train_y = data_y[{{1, max_size}}]            -- the rest are for training

train_x = data_x[{{1, (data_x:size(1)-val_size)}, {}}]            -- the rest are for training
train_y = data_y[{{1, (data_y:size(1)-val_size)}}]            -- the rest are for training

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