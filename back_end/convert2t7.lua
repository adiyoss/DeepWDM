----------------------------------------------------------------------
require 'torch'   -- torch
dofile ('utils.lua')
--require('mobdebug').start()
----------------------------------------------------------------------

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-x_filename', 'tmp.features', 'the path to the features file')
cmd:option('-y_filename', 'tmp.labels', 'the path to the labels file')
cmd:option('-input_dim', 39, 'the input dimension size')
cmd:option('-output_dim', 2, 'the output dimension size')
cmd:option('-output_x', 'x.t7', 'the t7 features path')
cmd:option('-output_y', 'y.t7', 'the t7 labels path')

cmd:text()
opt = cmd:parse(arg or {})

local x_filename = opt.x_filename
local y_filename = opt.y_filename
local data_x = load_data(x_filename, opt.input_dim)
local tmp_y = load_data(y_filename, opt.output_dim)
local data_y = tmp_y[{{}, 1}]

torch.save(opt.output_x, data_x)
torch.save(opt.output_y, data_y) 