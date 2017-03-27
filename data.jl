export data
module data

using MLDatasets

# Returns a tuple of two tuples: training and test data and labels
function cifar10(dir=nothing, onehot=true; dtype = Float32)
    dir = (dir == nothing) ? string(pwd(),"/cifar10") : dir
    loader = MLDatasets.CIFAR10
    (xtr, ytr) = loader.traindata(dir)
    (xts, yts) = loader.testdata(dir)
    xtr = convert(Array{dtype}, xtr)
    xts = convert(Array{dtype}, xts)
    if onehot
        ytr = toonehot(ytr+1, 10)
        yts = toonehot(yts+1, 10)
    end
    return ((xtr, ytr), (xts, yts))
end

function toonehot(ytrnraw, numclass; dtype=Float32)
    yonehot = zeros(dtype, numclass, length(ytrnraw))
    # println(ytrnraw)
    for (i, y) in enumerate(ytrnraw)
      # println(i," ", y)
      yonehot[y, i] = 1.0
   end
    #y[ytrnraw[:], 1:length(ytrnraw)] = 1.0
    return yonehot
end

end
