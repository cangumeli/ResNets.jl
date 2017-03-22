module ResNetCifar2
using Knet

export ResNetCifar2, loss, predict, init_params

# TODO: refactor this config to a mutatable file
global gpu = true
global dtype = Float32
global bmomentum = dtype(0.1)
global beps = .0001
global n = 18
global s = Any[]
global wdecay=.0001

function init_conv_param(height, width, input, output)
    w = randn(dtype, height, width, input, output) / dtype(sqrt(2.0 / (height * width * output)))
    if gpu
        w = convert(KnetArray{dtype}, w)
    end
    return w
end

function init_fc_param(input, output)
    w = randn(output, input) / dtype(sqrt(2.0 / output * input))
    b = zeros(output, 1)
    if gpu
        w = convert(KnetArray{dtype}, w)
        b = convert(KnetArray{dtype}, b)
    end
    return w, b
end

function init_batchnorm_param(channels)
    gamma = ones(dtype, 1, 1, channels, 1)
    beta = ones(dtype, 1, 1, channels, 1)
    if gpu
        gamma = convert(KnetArray{dtype}, gamma)
        beta = convert(KnetArray{dtype}, beta)
    end
    return gamma, beta
end

function init_params(;reset_stat=true)
    params = Any[]
    if reset_stat
        global s = Any[]
    end
    stats = s
    push!(params, init_conv_param(3, 3, 3, 16))
    for param in init_batchnorm_param(16)
        push!(params, param)
    end
    push!(stats, init_stat(16))
    for i=1:2n
        push!(params, init_conv_param(3, 3, 16, 16))
        for param in init_batchnorm_param(16)
            push!(params, param)
        end
        push!(stats, init_stat(16))
    end
    # Add bottleneck convolution and batchnorm
    push!(params, init_conv_param(1, 1, 16, 32))
    #=for param in init_batchnorm_param(32)
        push!(params, param)
    end=
    push!(stats, init_stat(32))=#
    # Add the first, different size convolution for the first layer for the next group
    push!(params, init_conv_param(3, 3, 16, 32))
    for param in init_batchnorm_param(32)
        push!(params, param)
    end
    push!(stats, init_stat(32))
    
    for i = 1:(2n-1)
        push!(params, init_conv_param(3, 3, 32, 32))
        for param in init_batchnorm_param(32)
            push!(params, param)
        end
        push!(stats, init_stat(32))
    end

    # Add the third group of convolutions
    push!(params, init_conv_param(1, 1, 32, 64))
    push!(params, init_conv_param(3, 3, 32, 64))
    for param in init_batchnorm_param(64)
        push!(params, param)
    end
    push!(stats, init_stat(64))
    for i = 1:(2n-1)
        push!(params, init_conv_param(3, 3, 64, 64))
        for param in init_batchnorm_param(64)
            push!(params, param)
        end
        push!(stats, init_stat(64))
    end
    # fcs = init_fc_param(16*16*16, 10)
    fcs = init_fc_param(4 * 4 * 64, 10)
    for f in fcs
        push!(params, f)
    end
    return params
end

function init_stat(depth)
    running_mean = gpu ? convert(KnetArray{dtype}, zeros(dtype, 1, 1, depth, 1)) : zeros(1, 1, depth, 1)
    running_var = gpu ? convert(KnetArray{dtype}, zeros(dtype, 1, 1, depth, 1)) : zeros(dtype, 1, 1, depth, 1)
    return Dict(:running_mean => running_mean, :running_var => running_var)
end

function sbatch_norm(gamma, beta, x, stats; mode=:train)
    assert(mode in [:train, :test])
    if mode == :train
        m = size(x, 1) * size(x, 2) * size(x, 4)
        mu = sum(x, (1, 2, 4)) ./ m
        x_mu = x .- mu
        var = sumabs2(x_mu, (1, 2, 4)) ./ m
        xnorm = x_mu ./ sqrt(var .+ beps)
        # println(size(mu), size(stats[:running_mean]))
        stats[:running_mean] = bmomentum .* mu  .+ (1 - bmomentum) .* stats[:running_mean]
        stats[:running_var] = bmomentum .* mu  .+ (1 - bmomentum) .* stats[:running_var]
        return gamma .* xnorm .+ beta
    end
    mu = stats[:running_mean]
    var = stats[:running_var]
    xnorm = (x .- mu) ./ sqrt(var .+ beps)
    return gamma .* xnorm .+ beta
end
              
# One residual layer for testing
function predict(w, x; mode=:train)
    # println(6n+2)
    x = conv4(w[1], x; padding=1)
    x = sbatch_norm(w[2], w[3], x, s[1])
    x = relu(x)
    res = x
    wstart=4
    for i = 1:2n
        x = conv4(w[wstart], x; padding=1)
        x = sbatch_norm(w[wstart+1], w[wstart+2], x, s[i+1]; mode=mode)
        x = relu(x)
        wstart += 3
        if (i % 2 == 0)
            x = x + res
            res = x
        end
    end

    res = conv4(w[wstart], res; stride=2) # bottleneck
    wstart += 1
    for i = 1:2n
        x = conv4(w[wstart], x; padding=1,
                  stride=1+(i==1))
        x = sbatch_norm(w[wstart+1], w[wstart+2], x, s[2n+i+1]; mode=mode)
        x = relu(x)
        wstart += 3
        if i % 2 == 0
            x = x + res
            res = x
        end
    end

    res = conv4(w[wstart], res; stride=2) # bottleneck
    wstart += 1
    for i = 1:2n
        x = conv4(w[wstart], x; padding=1,
                  stride=1+(i==1))
        x = sbatch_norm(w[wstart+1], w[wstart+2], x, s[4n+i+1]; mode=mode)
        x = relu(x)
        wstart += 3
        if i % 2 == 0
            x = x + res
            res = x
        end
    end
    x = pool(x; mode=2)
    return w[end-1] * mat(x) .+ w[end]
end

function loss(w, x, ygold; mode=:train)
    ypred = predict(w, x; mode=mode)
    ynorm = ypred .- log(sum(exp(ypred), 1))
    softloss = -sum(x-sum(ygold .* ynorm) / size(ygold, 2))
    regloss = 0.0
    for i = 1:3:(length(w)-2)
        regloss += wdecay .* sum(w[i].^2)
    end
    regloss += sum(wdecay .* w[end-1].^2)
    return softloss + regloss
end

lossgradient = grad(loss)

function accuracy(w, dtst)
    x, y = dtst
    println(typeof(x))
    ncorrect = 0
    println(gpu)
    for i = 1:100:size(x, 4)
        x_ = x[:, :, :, i:i+99]
        y_ = y[:, i:i+99]
        if gpu
            x_ = convert(KnetArray{dtype}, x_)
            y_ = convert(KnetArray{dtype}, y_)
        end
        ypred = predict(w, x_; mode=:test)
        
        ncorrect += sum(y_ .* (ypred .== maximum(ypred,1)))
    end
    # println(ncorrect)
    return ncorrect / size(x, 4)
end

function test()
    global n = 1
    w = init_params()
    #=for p in w
        println(size(p))
    end=#
    x = rand(dtype, 32, 32, 3, 128)
    x .-= mean(x)
    if gpu
        x = convert(KnetArray{Float32}, x)# rand(32, 32, 3, 128))
    end
    res = predict(w, x)
    println("\n\n")
    return res
    # return predict(w, x, 15, s)
end

end
