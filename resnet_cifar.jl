module ResNetCifar
using Knet

export ResNetCifar

type ResNet
    params::Any
    grads::Any
    n::Int64
    preactivation::Bool
end

function add_params!(model:: ResNet, params)
    for param in params
        push!(model.weights, param)
    end
end

param_config = Dict{Symbol, Any}()
param_config[:gpu] = true
param_config[:dtype] = Float32

# Initialize the parameters of convolutional filters
# https://arxiv.org/abs/1502.01852
function init_conv_params(height, width, input, output;
                          bnorm=true, mode=:back, preactivation=false)
    assert(mode in [:back, :forw])
    gpu = param_config[:gpu]
    dtype = param_config[:dtype]
    # initialize the weight
    stdev = dtype(sqrt(2.0 / (height * width * ((mode == :back) ? output : input))))
    w = randn(dtype, height, width, input, output) ./ stdev
    params = nothing
    if bnorm # Lasagne and TF are followed in 1-initialization of gamma
        gamma = ones(dtype, 1, 1, output, 1)
        beta = zeros(dtype, 1, 1, output, 1)
        params = preactivation ? [gamma, beta, w] : [w, gamma, beta]
    else # the bias is irrelevant in batch norm, just ignore it
        b = zeros(dtype, 1, 1, output, 1)
        params = [w, b]
    end
    if gpu
        params = map(param->convert(KnetArray{dtype}, param), params)
    end
    return params
end

function init_fc_params(height, width, input, output; bnorm=false)
    dtype = param_config[:dtype]
    w = randn(dtype, height, width) ./ dtype(sqrt(2.0 / (hight * width)))
    if bnorm
        error("BatchNorm for FC not implemented yet")
    else
        b = zeros(dtype, width, 1)
        params = [w, b]
    end
    if param_config[:gpu]
        params = map(param->convert(KnetArray{dtype}, param), params)
    end
    return params
end

# TODO: add preactivation setting
function create_model(n::Int, preactivation=false)
    model = ResNet(params=Any[], grads=Any[], n=n, preactivation=preactivation)
    add_params!(model, init_conv_params(3, 3, 3, 16))
    for i = 1:2n
        add_params!(model, init_conv_params(3, 3, 16, 16))
    end
    for i = 1:2n
        if i == 3 #bottleneck
            add_params!(model, init_conv_params(1, 1, 16, 32))
        end
        add_params!(model, init_conv_params(3, 3, 32, 32))
    end
    for i = 1:2n
        if i == 3 #bottleneck
            add_params!(model, init_conv_params(1, 1, 32, 64))
        end
        add_params!(model, init_conv_params(3, 3, 64, 64))
    end
    # the final fc layer
    add_params!(model, init_fc_params(2048, 10))
    return model
end

function batch_norm(input, gamma, beta; mode=:train)
    assert(mode in [:train, :test])
    # TODO: complete this implementation
end



end
