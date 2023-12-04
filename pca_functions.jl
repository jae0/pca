

function simulated_data(nobs=60)
    # generate data (The simulation is inspired by biological measurement of expression of genes in cells, and so you can think of the two variables as cells and genes):
    mu_1 = 10.0 * ones(nobs ÷ 3)
    mu_0 = zeros(nobs ÷ 3)
    s = I(nobs ÷ 3)
    mvn_0 = MvNormal(mu_0, s)
    mvn_1 = MvNormal(mu_1, s)

    # create a diagonal block like expression matrix, with some non-informative genes;
    # not all features/genes are informative, some might just not differ very much between cells)
    obs = transpose(
        vcat(
            hcat(rand(mvn_1, nobs ÷ 2), rand(mvn_0, nobs ÷ 2)),
            hcat(rand(mvn_0, nobs ÷ 2), rand(mvn_0, nobs ÷ 2)),
            hcat(rand(mvn_0, nobs ÷ 2), rand(mvn_1, nobs ÷ 2)),
        ),
    )

    df_obs = DataFrame(obs, :auto)
    df_obs[!, :id] = 1:nobs

    return obs, df_obs
end



@model function pPCA(x, ::Type{TV}=Array{Float64}) where {TV}

    nvar, nobs = size(x)

    # latent variable z
    z ~ filldist(Normal(), nvar, nobs)

    # weights/loadings W
    w ~ filldist(Normal(), nvar, nvar)

    # mean offset
    m ~ MvNormal(ones(nvar))
    mu = (w * z .+ m)'
    for d in 1:nvar
        x[d, :] ~ MvNormal(mu[:, d], ones(nobs))
    end
end;


@model function pPCA_ARD(x, ::Type{TV}=Array{Float64}) where {TV}

    nvar, nobs = size(x)

    # latent variable z
    z ~ filldist(Normal(), nvar, nobs)

    # weights/loadings w with Automatic Relevance Determination part
    alpha ~ filldist(Gamma(1.0, 1.0), nvar)
    w ~ filldist(MvNormal(zeros(nvar), 1.0 ./ sqrt.(alpha)), nvar)

    mu = (w' * z)'

    tau ~ Gamma(1.0, 1.0)
    for d in 1:nvar
        x[d, :] ~ MvNormal(mu[:, d], 1.0 / sqrt(tau))
    end
end;



@model function pPCA_residual(x, batch, ::Type{TV}=Array{Float64}) where {TV}
    nvar, nobs = size(x)

    # latent variable z
    z ~ filldist(Normal(), nvar, nobs)

    # weights/loadings w
    w ~ filldist(Normal(), nvar, nvar)

    # covariate vector
    w_batch = TV{1}(undef, nvar)
    w_batch ~ MvNormal(ones(nvar))

    # mean offset
    m = TV{1}(undef, nvar)
    m ~ MvNormal(ones(nvar))
    mu = m .+ w * z + w_batch .* batch'

    for d in 1:nvar
        x[d, :] ~ MvNormal(mu'[:, d], ones(nobs))
    end
end;

 
## remainder are householder variation copied and modified from
# https://github.com/jae0/HouseholderBPCA/blob/master/ubpca_improved.ipynb
# source: https://github.com/jae0/HouseholderBPCA/blob/master/py_stan_code/ppca_house_improved.stan


function sign_convention(U)
    nq,_ = size(U)
    for q in 1:nq
        if U[1,q] < 0
            U[:,q] *= -1.0
        end
    end
    return U
end


function pca_standard(Y, type="cor")
    # Y= X'
    if type=="cor"
        C = cor(Y)
    else
        C = cov(Y)
    end

    E = svd(C)
    U = sign_convention(E.U)  # eigenvectors
    sigma_pca = sqrt.(E.S)   #sqrt(eigenvalues)
    PC = Y * U 
    return U, sigma_pca, C, PC
end

# function V_low_tri_plus_diag(nq::Int, V)
#     for q in 1:nq
#         V[:, q] = V[:, q] ./ sqrt(sum(V[:, q] .^ 2))
#     end
#     return (V)
# end


# function H_prod_right(V)
#     nvar, nq = size(V)
#     H_prod = zeros(Real, nvar, nvar, nq + 1)
#     H_prod[:, :, 1] = Diagonal(repeat([1.0], nvar))

#     for q in 1:nq
#         H_prod[:, :, q + 1] = Householder(nq - q + 1, V) * H_prod[:, :, q]
#     end
#     return (H_prod)
# end
 

function Householder(k::Int, V)
    v = V[:, k]
    sgn = sign(v[k])
    v[k] += sgn
    H = I - (2.0 / dot(v, v) * (v * v'))
    H[k:end, k:end] = -1.0 * sgn .* H[k:end, k:end]
    return (H)
end


function Householder_invert(v, nvar)
    nq = length(v)
    sgn = sign(v[1])  
    u = v .+ sgn * norm(v)*I(nq)[:,1]
    u = u / norm(u)
    i = (nvar - nq + 1) : nvar
    H = zeros( nvar, nvar) + I(nvar)
    H[i, i] = -1.0 .* sgn .* ( I(nq) .- 2.0 .* u*u') 
    return (H)
end


function eigenvector_to_householder(U, nq)  
    # to convet eigenvectors U to householder transformed v (for initializing Turing) 
    nvar, _ = size(U)
    HU = U[:, 1:nq]
    # HU = reverse(U, dims=2)  # used for stan 
    v_mat = zeros(size(HU))
    for q in 1:nq
        v_mat[q:nq, q] = HU[q:nq, q] 
        HU = Householder_invert(HU[q:nq, q], nvar) * HU 
    end
    return v_mat[tril!(trues(size(v_mat)))]  # flatten
end


function householder_to_eigenvector(nvar::Int64, nq::Int64, V)
      
    for q in 1:nq
        V[:, q] = V[:, q] ./ sqrt(sum(V[:, q] .^ 2))
    end

    H_prod = zeros(Real, nvar, nvar, nq + 1)
    H_prod[:, :, 1] = Diagonal(repeat([1.0], nvar))

    for q in 1:nq
        k = nq - q + 1
        H_prod[:, :, q + 1] = Householder(k, V) * H_prod[:, :, q]
    end

    return H_prod[:, 1:nq, nq + 1]
end


 



 
@model function PCA_BH_model(x, nq::Int, ::Type{T}=Float64; error=1e-9 ) where {T}
    # householder, modified to be all in one
    nvar, nobs = size(x)

    @assert nq <= nvar

    # parameters
    sigma_noise ~ LogNormal(0.0, 0.5)

    # currently, Bijectors.ordered is broken, revert for better posteriors once it works again
    # sigma ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nq) )) )  
    sigma ~  MvLogNormal(MvNormal(ones(nq) ))
    
    v ~ filldist(Normal(0.0, 1.0), Int(nvar * nq - nq * (nq - 1) / 2))
    
    v_mat = zeros(T, nvar, nq)
    v_mat[tril!(trues(size(v_mat)))] .= v
    U = householder_to_eigenvector(nvar, nq, v_mat)

    W = zeros(T, nvar, nq)
    W += U * Diagonal(sigma)

    # extra careful make sure positive definite
    Kmat = zeros(T, nvar, nvar)
    Kmat += W * W'  
    diag_err = sigma_noise^2 + error
    for d in 1:nvar
        Kmat[d, d] = Kmat[d, d] + diag_err
    end
    L = LinearAlgebra.cholesky(Kmat).L

    for q in 1:nq
        r = sqrt.(norm(v_mat[:, q]))
        r ~ Gamma(2.0, 2.0)  # new .. Gamma in stan is same as in Distributions
        Turing.@addlogprob! (-log(r) * (nvar - q))
    end

    if minimum(sigma) < error 
        Turing.@addlogprob! Inf
        return
    end

    Turing.@addlogprob! -0.5 * sum(sigma .^ 2) + (nvar - nq - 1) * sum(log.(sigma))
    for qi in 1:nq
        for qj in (qi + 1):nq
            Turing.@addlogprob! log(sigma[nq - qi + 1]^2) - sigma[nq - qj + 1]^2
        end
    end
    Turing.@addlogprob! sum(log.(2.0 * sigma))

    L_full = zeros(T, nvar, nvar)
    L_full += L * transpose(L)
    
    # make symmetric
    for d in 1:nvar
        for k in (d + 1):nvar
            L_full[d, k] = L_full[k, d]
        end
    end
    
    x ~ filldist(MvNormal(L_full), nobs)
    return 
end


  


function PCA_BH_extract( chain, X, nq, return_object  )
 
    nvar, nobs = size(X)
    n_samples,_,_ = size(chain) 
 
    if return_object == :scores
        # full posteriors
        sigma = PCA_BH_extract( chain, X, nq, return_object="sigma" )  # posteriors
        U = PCA_BH_extract( chain, X, nq, return_object="eigenvectors" )  # posteriors

        scores = Array{Float64}(undef, nq, nobs, n_samples)
        for i in 1:n_samples
            W = U[i,:, :] * (LinearAlgebra.I(nq) .* sigma[:, i])
            scores[:, :, i] = W' * X 
        end
        return scores
    end

    
    if return_object == :scores_mean
        scores = PCA_BH_extract( chain, X, nq, return_object="scores" )
        scores_mean = DataFrame( convert(Array{Float64}, mean(scores, dims=3)[:,:,1])', :auto)
        rename!(scores_mean, Symbol.(["pc" * string(i) for i in collect(1:nq)]))
        return scores_mean
    end


    if return_object == :sigma
        # full posteriors
        ss = collect(get(chain, [:sigma]).sigma)
        sigma = zeros(nq, n_samples)
        for d in 1:nq
            sigma[d, :] = Array(ss[d])  # nobs X 1 array
        end
        return sigma
    end


    if return_object == :scores_mean
        sigma = PCA_BH_extract( chain, X, nq, return_object="sigma" )
        scores_mean = DataFrame( convert(Array{Float64}, mean(scores, dims=3)[:,:,1])', :auto)
        rename!(sigma_mean, Symbol.(["pc" * string(i) for i in collect(1:nq)]))
        return sigma_mean
    end


    if return_object == :eigenvalues
        eigenvalues = PCA_BH_extract( chain, X, nq, return_object="sigma" )  # posteriors
        return eigenvalues
    end

    
    if return_object == :eigenvectors
        # unscaled 
        vv = collect(get(chain, [:v]).v)
        V = zeros(Real, nvar, nq)
        vv_mat = zeros(Float64, n_samples, nvar, nq)
    
        for i in 1:n_samples
            index = BitArray(zeros(n_samples, nvar, nq))
            index[i, :, :] = tril!(trues(size(V)))
            tmp = zeros(size(vv)[1])
            for j in 1:(size(vv)[1])
                tmp[j] = vv[j][i]
            end
            vv_mat[index] .= tmp
        end
     
        U = zeros(Float64, n_samples, nvar, nq)
        for i in 1:n_samples
            U[i,:,:] = householder_to_eigenvector(nvar, nq, vv_mat[i, :, :])
        end

        return U

    end

end
