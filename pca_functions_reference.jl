 

# Householder transform (rotation invariant form of pPCA
# reference (unmodified) code from: https://turing.ml/v0.21/tutorials/11-probabilistic-pca

function V_low_tri_plus_diag(Q::Int, V)
    for q in 1:Q
        V[:, q] = V[:, q] ./ sqrt(sum(V[:, q] .^ 2))
    end
    return (V)
end

function Householder(k::Int, V)
    v = V[:, k]
    sgn = sign(v[k])
    v[k] += sgn
    H = LinearAlgebra.I - (2.0 / dot(v, v) * (v * v'))
    H[k:end, k:end] = -1.0 * sgn .* H[k:end, k:end]
    return (H)
end

function H_prod_right(V)
    D, Q = size(V)

    H_prod = zeros(Real, D, D, Q + 1)
    H_prod[:, :, 1] = Diagonal(repeat([1.0], D))

    for q in 1:Q
        H_prod[:, :, q + 1] = Householder(Q - q + 1, V) * H_prod[:, :, q]
    end

    return (H_prod)
end

function orthogonal_matrix(D::Int64, Q::Int64, V)
    V = V_low_tri_plus_diag(Q, V)
    H_prod = H_prod_right(V)
    return (H_prod[:, 1:Q, Q + 1])
end


@model function pPCA_householder_reference(x, K::Int, ::Type{T}=Float64) where {T}
    D, N = size(x)
    @assert K <= D

    # parameters
    sigma_noise ~ LogNormal(0.0, 0.5)
    v ~ filldist(Normal(0.0, 1.0), Int(D * K - K * (K - 1) / 2))
    sigma ~ Bijectors.ordered(MvLogNormal(MvNormal(ones(K))))

    v_mat = zeros(T, D, K)
    v_mat[tril!(trues(size(v_mat)))] .= v
    U = orthogonal_matrix(D, Q, v_mat)

    W = zeros(T, D, K)
    W += U * Diagonal(sigma)

    Kmat = zeros(T, D, D)
    Kmat += W * W'
    for d in 1:D
        Kmat[d, d] = Kmat[d, d] + sigma_noise^2 + 1e-12
    end
    L = LinearAlgebra.cholesky(Kmat).L

    for q in 1:Q
        r = sqrt.(sum(dot(v_mat[:, q], v_mat[:, q])))
        Turing.@addlogprob! (-log(r) * (D - q))
    end

    Turing.@addlogprob! -0.5 * sum(sigma .^ 2) + (D - Q - 1) * sum(log.(sigma))
    for qi in 1:Q
        for qj in (qi + 1):Q
            Turing.@addlogprob! log(sigma[Q - qi + 1]^2) - sigma[Q - qj + 1]^2
        end
    end
    Turing.@addlogprob! sum(log.(2.0 * sigma))

    L_full = zeros(T, D, D)
    L_full += L * transpose(L)
    # fix numerical instability (non-posdef matrix)
    for d in 1:D
        for k in (d + 1):D
            L_full[d, k] = L_full[k, d]
        end
    end

    return x ~ filldist(MvNormal(L_full), N)

end

#=

# Dimensionality of latent space
Random.seed!(1789);
Q = 2
n_samples = 700
ppca_householder = pPCA_householder(Matrix(dat)', Q)
chain_ppcaHouseholder = sample(ppca_householder, NUTS(), n_samples);

# Extract mean of v from chain
N, D = size(dat)
vv = mean(group(chain_ppcaHouseholder, :v))[:, 2]
v_mat = zeros(Real, D, Q)
v_mat[tril!(trues(size(v_mat)))] .= vv
sigma = mean(group(chain_ppcaHouseholder, :sigma))[:, 2]
U_n = orthogonal_matrix(D, Q, v_mat)
W_n = U_n * (LinearAlgebra.I(Q) .* sigma)

# Create array with projected values
z = W_n' * transpose(Matrix(dat))
df_post = DataFrame(convert(Array{Float64}, z)', :auto)
rename!(df_post, Symbol.(["z" * string(i) for i in collect(1:Q)]))
df_post[!, :sample] = 1:n
df_post[!, :species] = species

@vlplot(:point, x = :z1, y = :z2, color = "species:n")(df_post)


## Create data projections for each step of chain
vv = collect(get(chain_ppcaHouseholder, [:v]).v)
v_mat = zeros(Real, D, Q)
vv_mat = zeros(Float64, n_samples, D, Q)
for i in 1:n_samples
    index = BitArray(zeros(n_samples, D, Q))
    index[i, :, :] = tril!(trues(size(v_mat)))
    tmp = zeros(size(vv)[1])
    for j in 1:(size(vv)[1])
        tmp[j] = vv[j][i]
    end
    vv_mat[index] .= tmp
end

ss = collect(get(chain_ppcaHouseholder, [:sigma]).sigma)
sigma = zeros(Q, n_samples)
for d in 1:Q
    sigma[d, :] = Array(ss[d])
end

samples_raw = Array{Float64}(undef, Q, N, n_samples)
for i in 1:n_samples
    U_ni = orthogonal_matrix(D, Q, vv_mat[i, :, :])
    W_ni = U_ni * (LinearAlgebra.I(Q) .* sigma[:, i])
    z_n = W_ni' * transpose(Matrix(dat))
    samples_raw[:, :, i] = z_n
end

# initialize a 3D plot with 1 empty series
plt = plot(
    [100, 200, 300];
    xlim=(-4.00, 7.00),
    ylim=(-100.00, 0.00),
    group=["Setosa", "Versicolor", "Virginica"],
    markercolor=["red", "blue", "black"],
    title="Visualization",
    seriestype=:scatter,
)

anim = @animate for i in 1:n_samples
    scatter!(
        plt,
        samples_raw[1, 1:50, i],
        samples_raw[2, 1:50, i];
        color="red",
        seriesalpha=0.1,
        label="",
    )
    scatter!(
        plt,
        samples_raw[1, 51:100, i],
        samples_raw[2, 51:100, i];
        color="blue",
        seriesalpha=0.1,
        label="",
    )
    scatter!(
        plt,
        samples_raw[1, 101:150, i],
        samples_raw[2, 101:150, i];
        color="black",
        seriesalpha=0.1,
        label="",
    )
end
gif(anim, "anim_fps.gif"; fps=5)

# kernel density estimate
using KernelDensity
dens = kde((vec(samples_raw[1, :, :]), vec(samples_raw[2, :, :])))
StatsPlots.plot(dens)

=#


  
function phPCA_extract( chain; return_object="scores"  )
 
    if return_object=="scores" 
        
        sigma = mean(group(chain, :sigma))[:, 2]
        v = mean(group(chain, :v))[:, 2]
        v_mat = zeros(Real, D, Q)
        v_mat[tril!(trues(size(v_mat)))] .= v
        U = orthogonal_matrix(D, Q, v_mat)
    
        W = zeros(Real, D, Q)
        W += U * Diagonal(sigma)
    
        # Create array with projected values
        z = W' * X
        df_post = DataFrame(convert(Array{Float64}, z)', :auto)
        rename!(df_post, Symbol.(["z" * string(i) for i in collect(1:Q)]))
        return df_post 
    end

end

  
