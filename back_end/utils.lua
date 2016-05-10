-- load from txt file
function load_data(path, input_dim)    
  local num_of_examples = get_num_of_lines(path)
  local file = io.open(path, "r")
  local data = torch.zeros(num_of_examples, input_dim)
  
  if file then
    local i = 1
    for line in file:lines() do      
      local j = 1
      for str in string.gmatch(line, "(%S+)") do
        data[i][j] = str
        j = j + 1
      end
      i = i + 1
    end
  else
    print("\n==>ERROR: can not open file.")    
  end  
  if not file then
    file:close()
  end
  return data
end

function get_mini_batch(x, y, rho)  
  mini_batch_idx = mini_batch_idx + 1
  start_idx = (mini_batch_idx-1)*rho + 1
  end_idx = (mini_batch_idx)*rho

  -- validation
  if x:size(1) < end_idx then 
    mini_batch_idx = 1
    start_idx = 1
    end_idx = rho
  end
  
  local mini_batch_x = x[{{start_idx, end_idx}, {}}]
  local mini_batch_y = y[{{start_idx, end_idx}}]
  
  return mini_batch_x, mini_batch_y
end

function save_tensor_2_txt(t, output_path)
  fid = io.open(output_path, 'w')
  for i=1, t:size(1) do
    fid:write(t[i] .. '\n')
  end
  fid:close()
end

-- argmax.lua
local function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

local function argmax_2D(matrix)
   local nRows = matrix:size(1)
   local result = torch.Tensor(nRows)
   for i = 1, nRows do
      result[i] = argmax_1D(matrix[i])
   end
   return result
end

-- index of largest element
-- ARGS:
-- tensor : 1D or 2D Tensor
-- RETURNS:
-- result : scalar (if v is 1D) or 1D Tensor (if v is 2D)
--          if scalar i : integer in [1, v:size(1)] such that v[i] >= v[k] for all k
--          if 1D Tensor, then the scalar i for each row
function argmax(tensor)
   local nDimension = tensor:nDimension()
   if nDimension == 1 then
      return argmax_1D(tensor)
   elseif nDimension == 2 then
      return argmax_2D(tensor)
   else
      error('tensor has %d dimensions, not 1 or 2', nDimension)
   end
end


local function isempty(s)
  return s == nil or s == '' or s == '\n'
end


function get_num_of_lines(path)
  local i = 0
  local file = io.open(path, "r")
  if file then
    for line in file:lines() do
      if isempty(line) ~= true then
        i = i + 1
      end
    end
  else
    print("\n==>ERROR: can not open file.")    
  end  
  if not file then
    file:close()
  end
  return i
end