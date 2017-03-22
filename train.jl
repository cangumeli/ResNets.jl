include("resnet_cifar2.jl")
include("data.jl")
include("imgproc.jl")
using Knet

function main()
    dtrn, dval, dtst = loaddata()
    train(dtrn, dval)
end

function loaddata(;nval=5000, ms=true, ms_mode=:pixel,  augment=true)
    tr, ts = Data.cifar10()
    (xts, yts) = ts
    # Train-validation split
    (xtr, ytr) = tr
    ntrain = size(xtr, 4)
    sample = Array{Int32}(1:ntrain)
    shuffle!(sample)
    (xval, yval) = (xtr[:, :, :, sample[(ntrain-nval+1):ntrain]], ytr[:, sample[(ntrain-nval+1):ntrain]])
    (xtr, ytr) = (xtr[:, :, :, sample[1:(ntrain-nval)]], ytr[:, sample[1:(ntrain-nval)]])
    # Data preprocessing
    if ms
        println(size(xtr))
        mn = ImgProc.mean_subtract!(xtr; mode=ms_mode)
        println(size(mn))
        println(size(xts))
        xts .-= mn
        xval .-= mn
    end
    if augment # add padding for virtual augmentation
        function augment_data(x)
            dims = size(x)
            d1, d2 = dims[1], dims[2]
            tr = zeros(Float32, (dims[1]+4, dims[2]+4, dims[3:end]...))
            tr[3:(d1+2), 3:(d2+2), :, :] = x
            return tr
        end
        xtr = augment_data(xtr)
        xts = augment_data(xts)
        xval = augment_data(xval)
    end
    return ((xtr, ytr), (xval, yval), (xts, yts))
end

function next_batch(dtrn; bsize=128, pflip=0.5, augmented=true)
    xtr, ytr = dtrn
    ntrain = size(xtr, 4)
    sample = rand(1:ntrain, bsize)
    if !augmented # TODO: add gpu support here too!
        return xtr[:, :, :, sample], ytr[:, sample]
    end
    batch = zeros(32, 32, 3, bsize)
    for (i, j) in enumerate(sample)
        start = rand(1:5, 2)
        r, c = start[1], start[2]
        x =  xtr[:, :, :, j]
        x_ = x[r:r+31, c:c+31, :, :]
        if rand() < pflip # random horizontal flip
            x_ = x_[:, end:-1:1, :, :]
        end
        batch[:, :, :, i] = x_
    end
    y = ytr[:, sample]
    if gpu
        batch = convert(KnetArray{ResNetCifar2.dtype}, batch)
        y = convert(KnetArray{ResNetCifar2.dtype}, y)
    end
    return batch, y
end

function init_model()
    w = ResNetCifar2.init_params()
    g = ResNetCifar2.lossgradient
    w, g
end

function train(dtrn, dval; num_iters=100000, initlr=0.01, accuracy_period=10000)
    w, grad = init_model()
    lr = initlr
    println("Accuracy ", ResNetCifar2.accuracy(w, dval))
    for i = 1:num_iters
        (x, y) = next_batch(dtrn)
        dw = grad(w, x, y)
        for j = 1:length(dw)
            axpy!(-lr, dw[j], w[j])
        end
        if i == 400
            lr *= 10
        end
        if i % print_period == 0
           println("Accuracy ", ResNetCifar2.accuracy(w, dval))
        end
    end
end

main()
