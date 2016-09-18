----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Loss function definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll')
   cmd:option('-output_dim', 2, 'the output size')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

-- 2-class problem
noutputs = opt.output_dim
weights = torch.zeros(opt.output_dim)
weights[1] = 1
weights[2] = 1

----------------------------------------------------------------------
print '==> define loss'
if opt.loss == 'nll' then  
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion(weights))
else
   error('unknown -loss')
end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)