using Flux
using NeuralOperators
using HDF5
using Plots
using FFTW
using LinearAlgebra

const NO = NeuralOperators


# Import the first instance of my dataset.
# Visualize the dataset.
filename = "test/Frequencies/sample"
file = h5open(filename,"r")
f_i = read(file["1"])
u_i = read(file["500"])


f1 = heatmap(u_i,title="Original")

# Setup
x = 0.0:(1.0)/200:1.0
y = 0.0:(1.0)/200:1.0

X = reshape([xi for xi in x for _ in y],(length(x),length(y)))
Y = reshape([yi for _ in x for yi in y],(length(x),length(y)))

nx = length(x)
ny = length(y)

xdata = Array{Float64,4}(undef,3,nx,ny,1)
ydata = Array{Float64,4}(undef,1,nx,ny,1)

xdata[1,:,:,1] = u_i
xdata[2,:,:,1] = X
xdata[3,:,:,1] = Y
ydata[1,:,:,1] = u_i

### Show the fourier modes for the current implementation.
op = OperatorKernel(3=>3, (12,12),FourierTransform,identity,permuted=true)
conv = op.conv

xdata = permutedims(xdata,(2,3,1,4))

function apply_pattern(𝐱_truncated, 𝐰)
    x_size = size(𝐱_truncated) # [m.modes..., in_chs, batch]

    𝐱_flattened = reshape(𝐱_truncated, :, x_size[(end - 1):end]...) # [prod(m.modes), in_chs, batch], only 3-dims
    # 𝐱_weighted = einsum(𝐱_flattened, 𝐰) # [prod(m.modes), out_chs, batch], only 3-dims
    𝐱_shaped = reshape(𝐱_flattened, x_size[1:(end - 2)]..., size(𝐱_flattened)[2:3]...) # [m.modes..., out_chs, batch]

    return 𝐱_shaped
end


trans = FourierTransform((12,12))
s1 = NO.transform(trans,xdata)
s2 = NO.truncate_modes(trans,s1)
s3 = apply_pattern(s2,conv.weight)
s4 = NO.pad_modes(s3,(size(s1)[1:(end-2)]...,size(s3)[(end-1):end]...))
s5 = NO.inverse(trans,s4, size(xdata))

### Reimplemnt kernels

function box(modes,s)
    a = []
    for (i,mode) in enumerate(reverse(modes))
        push!(a,(s[i]-mode+1):1:(s[i]+mode))
    end 
    return a
end 

function truncate_modes(ft::FourierTransform, 𝐱_fft::AbstractArray)
    for i=1:size(𝐱_fft)[end-1]
        for j=1:size(𝐱_fft)[end]
            𝐱_fft[:,:,i,j] .= fftshift(𝐱_fft[:,:,i,j])
        end
    end 
    s = floor.(Int,size(𝐱_fft)[1:end-2]./2)
    view(𝐱_fft,box(floor.(Int,ft.modes./2),s)...,:,:)
end 

bpad_modes(𝐱::AbstractArray, dims::NTuple) = bpad_modes!(similar(𝐱, dims), 𝐱)

function bpad_modes!(𝐱_padded::AbstractArray, 𝐱::AbstractArray)
    fill!(𝐱_padded, eltype(𝐱)(0)) # zeros(eltype(𝐱), dims)
    modes = floor.(Int,size(𝐱)[1:end-2]./2)
    s = floor.(Int,size(𝐱_padded)[1:end-2]./2)
    shape = box(modes,s)

    for i=1:size(𝐱_padded)[end-1]
        for j=1:size(𝐱_padded)[end]
            𝐱_padded[shape...,i,j] .= 𝐱[map(d->1:d,size(𝐱)[1:end-2])...,i,j]
        end 
    end 

    return 𝐱_padded
end

function inverse(ft::FourierTransform, 𝐱_fft::AbstractArray{T, N},M::NTuple{N, Int64}) where {T, N}
    for i=1:size(𝐱_fft)[end-1]
        for j=1:size(𝐱_fft)[end]
            𝐱_fft[:,:,i,j] .= ifftshift(𝐱_fft[:,:,i,j])
        end
    end 
    return real.(irfft(𝐱_fft, M[1], 1:ndims(ft))) # [size(x_fft)..., out_chs, batch]
end

### Retest and make sure that central frequencies are accounted for.
r1 = transform(trans,xdata)
r2 = truncate_modes(trans,r1)
r3 = apply_pattern(r2,conv.weight)
r4 = bpad_modes(r3,(size(s1)[1:(end-2)]...,size(s3)[(end-1):end]...))
r5 = inverse(trans,r4,size(xdata))


# Compare each iteration step by step
c1 = heatmap(abs.(s1[:,:,1,1]),title="Full",c=:jet)
d1 = heatmap(abs.(r1[:,:,1,1]),title="Full - S",c=:jet)

c2 = heatmap(abs.(s1[:,:,1,1]),title="Full",c=:jet)
d2 = heatmap(abs.(fftshift(r1[:,:,1,1])),title="Full Transformed",c=:jet)

c3 = heatmap(abs.(s4[:,:,1,1]),title="Truncated - padded",c=:jet)
d3 = heatmap(abs.(r4[:,:,1,1]),title="Truncated - padded - S",c=:jet)

c4 = heatmap(abs.(s3[:,:,1,1]),title="Truncated",c=:jet)
d4 = heatmap(abs.(r3[:,:,1,1]),title="Truncated - S",c=:jet)

c5 = heatmap(abs.(s4[:,:,1,1]),title="Truncated - padded",c=:jet)
d5 = heatmap(abs.(ifftshift(r4[:,:,1,1])),title="Truncated - padded - S",c=:jet)

plot(c1,d1,c2,d2,c4,d4,c3,d3,c5,d5,layout=(5,2),size=(1000,1000))
savefig("test/Frequencies/steps.png")

## Reconstructed plots
f1 = heatmap(s5[:,:,1,1],title="Trivial truncation")
f2 = heatmap(r5[:,:,1,1],title="Centered truncation")
f3 = heatmap(u_i,title="Original")
plot(f1,f2,f3,size=(1000,1000))
savefig("test/Frequencies/reconstruction.png")
