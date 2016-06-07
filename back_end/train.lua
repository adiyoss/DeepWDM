----------------------------------------------------------------------
require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'ADAGRAD', 'optimization method: SGD | ADAM | ADAGRAD')
   cmd:option('-learningRate', 0.01, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters, gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ADAM' then
   optimState = {
      learningRate = opt.learningRate
   }
   optimMethod = optim.adam
   
elseif opt.optimization == 'ADAGRAD' then
   optimState = {
      learningRate = opt.learningRate,
   }
   optimMethod = optim.adagrad

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()
  
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,#rnnTrainData do
      -- disp progress
      xlua.progress(t, #rnnTrainData)

      --[[
      -- create mini batch
      local inputs = {}
      local targets = {}
      for step=1, opt.batchSize do
         input, target = get_mini_batch(trainData.data, trainData.labels, opt.rho)
         -- load new sample
         if opt.type == 'double' then
            input = input:double()            
         elseif opt.type == 'cuda' then
            input = input:cuda()
         end
         table.insert(inputs, input)
         table.insert(targets, target)
      end
      ]]--
      local inputs = rnnTrainData[t]
      local targets = rnnTrainLabels[t]
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- estimate f
                       local output = model:forward(inputs)
                       local err = criterion:forward(output, targets)
                       f = f + err

                       -- estimate df/dW
                       local df_do = criterion:backward(output, targets)
                       model:backward(inputs, df_do)
                      
                       for i=1,#targets do
                          -- update confusion
                          confusion:add(output[i], targets[i])
                       end  
                       
                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f, gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end