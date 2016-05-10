----------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
dofile 'utils.lua'

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function validate()
   -- local vars
   local time = sys.clock()
   local err = 0
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   -- test over test data
   print('==> compute loss on validation set:')
   for t = 1,valData:size() do
      -- disp progress
      xlua.progress(t, valData:size())      
      
      local inputs, targets = {}, {}      
      table.insert(inputs, valData.data[t])
      table.insert(targets, valData.labels[t])
      
      -- test sample
      local pred = model:forward(inputs)
      err = err + criterion:forward(pred, targets)
      
      confusion:add(pred[1], targets[1])    
   end

   -- timing
   time = sys.clock() - time
   time = time / valData:size()
   print("\n==> time to validate 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
   return err / valData:size()
end