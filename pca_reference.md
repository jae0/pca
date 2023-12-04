
# PCA basics copied from various sources 


## set up environment and test data

```julia
    # load helper files
    project_directory = joinpath( homedir(), "bio", "pca"  )
    include( joinpath( project_directory, "startup.jl" ))     
    include( joinpath( project_directory, "pca_functions.jl" ))     

    # Set a seed for reproducibility.
    Random.seed!(1);

    # dataset available in RDatasets
    Xdata = dataset("datasets", "iris")
    Xdata.id = Xdata.Species
 
    use_simulated_data = false
    if use_simulated_data
        _, Xdata = simulated_data(50)
        Xdata = Xdata[:,1:4]
        Xdata.id .= 1:50
    end

    X = Matrix(Xdata[!, 1:4])'
    # [[NOTE:: X is transposed relative ecology]]

    centerdata = true
    if centerdata
        Xmean = mean(X, dims=2)
        Xstd = std(X, dims=2)
        X = (X .- Xmean) ./ Xstd
    end

    nvar, nobs = size(X) # number of variables, number of measurements (observations) 
    nq = 2  # no latent factors
    n_samples = 1000

    @vlplot( :rect, x = "id:o", color = :value, encoding = {
        y = {field = "variable", type = "nominal", sort = "-x", axis = {title = "data"}}
        } )( DataFrames.stack(Xdata, 1:nvar) )
      

```


## Classical PCA

Projection of data $Y$ to lower dimensional latent space $X$:
$$\boldsymbol{Y} \in \R^{m\times n} \quad \rightarrow \quad \boldsymbol{X} \in \R^{m\times q}$$
usually by eigen-decomposition or SVD. 

https://juliastats.org/MultivariateStats.jl/dev/pca/
https://multivariatestatsjl.readthedocs.io/en/stable/pca.html

```julia

    # ~ simple PCA on corr matrix (as X is centered and standardized)

    M = fit(PCA, X; method=:svd, pratio=1, maxoutdim=nq )
    # Mk = fit(KernelPCA, X; maxoutdim=nq )
    # Mp = fit(PPCA, X; method=:bayes, maxoutdim= nq)
    Mm = fit(MDS, X; distances=false)

    # proj[:,1] = weights / "loadings" for PC1, etc
    proj = projection(M) 
     
    principalvars(M) ./ tvar(M) # contributions of each principal component towards explaining the total variance,
    
    # PCAscores = MultivariateStats.transform(M, X)
    (PCAscores = projection(M)' * (X .- mean(M)))

    using Plots
    h = plot(PCAscores[1,:], PCAscores[2,:], seriestype=:scatter, label="")
    plot!(xlabel="PC1", ylabel="PC2", framestyle=:box) # A few formatting options

    for i=1:4; plot!([0,proj[i,1]], [0,proj[i,2]], arrow=true, label=names(Xdata)[i], legend=:bottomleft); end
    display(h)
     

    # split half to training set
    Xtr = X[:,1:2:end]
    Xtr_labels = Vector(Xdata.id[1:2:end])

    # split other half to testing set
    Xte = X[:,2:2:end]
    Xte_labels = Vector(Xdata.id[2:2:end])

    Xte = (Xte .- Xmean) ./ Xstd
    Xtr = (Xtr .- Xmean) ./ Xstd
    
    # Suppose Xtr and Xte are training and testing data matrix, with each observation in a column. We train a PCA model, allowing up to 3 dimensions:

    M = fit(PCA, Xtr; maxoutdim=3 )
    loadings(M)
    projection(M)

    # Then, apply PCA model to the testing set

    Yte = predict(M, Xte)

    # reconstruct testing observations (approximately) to the original space

    Xr = reconstruct(M, Yte)

    setosa = Yte[:,Xte_labels.=="setosa"]
    versicolor = Yte[:,Xte_labels.=="versicolor"]
    virginica = Yte[:,Xte_labels.=="virginica"]

    p = scatter(setosa[1,:],setosa[2,:],setosa[3,:],marker=:circle,linewidth=0)
    scatter!(versicolor[1,:],versicolor[2,:],versicolor[3,:],marker=:circle,linewidth=0)
    scatter!(virginica[1,:],virginica[2,:],virginica[3,:],marker=:circle,linewidth=0)
    plot!(p,xlabel="PC1",ylabel="PC2",zlabel="PC3")


    # NOTE: can also compute using multivariatestats.jl with 
    # Compute probabilistic PCA using a Bayesian algorithm for a given sample covariance matrix S.
    S = cov(X')
    m =  mean(X, dims=2)   #means of each variable
    nvar, nobs =  size(X)

    Mbp = bayespca(S, vec(m), nvar )

    loadings(Mbp)
    projection(Mbp)

    # ? not sure, must verify
```


## Probabilitic PCA

Identifcation of the generative model that maps the latent space to data. I.e., the inverse operation:
$$\boldsymbol{Y} \in \R^{m\times n} \quad \leftarrow \quad \boldsymbol{X} \in \R^{m\times q}$$
such that:

$$\boldsymbol{Y} = \boldsymbol{X} \boldsymbol{W}^T + \boldsymbol{\epsilon}$$
$$\boldsymbol{X} \sim \text{N} (\boldsymbol{0}, \boldsymbol{I})$$ $$\boldsymbol{\epsilon} \sim \text{N} (\boldsymbol{0},\sigma^2 \boldsymbol{I})$$
When $\sigma=0$ this is the classical PCA. The weights $W$ are the parameters to be estimated, such that:

$$p(\boldsymbol{Y} | \boldsymbol{W}) = \prod_{i=1}^{m} \text{N} (\boldsymbol{Y}_{i,.}|0, \boldsymbol{W} \boldsymbol{W}^T + \sigma^2 \boldsymbol{I})$$
$$\boldsymbol{W} \boldsymbol{R} \boldsymbol{R}^T \boldsymbol{W}^T  = \boldsymbol{W} \boldsymbol{W}^T$$

But this creates rotationally symmetric solutions. 

```julia 
  
    M = pPCA(X)
    chain = sample(M, NUTS(), 100);

    # Extract parameter estimates for plotting - mean of posterior
    w = reshape(mean(group(chain, :w))[:, 2], (nvar, nvar))
    z = permutedims(reshape(mean(group(chain, :z))[:, 2], (nvar, nobs)))'
    mu = mean(group(chain, :m))[:, 2]

    PCAscores = w * z
    df_rec = DataFrame(PCAscores', :auto)
    df_rec[!, :id] = 1:nobs

    @vlplot(
        :rect,
        x = "id:o",
        color = :value,
        encoding = {
            y = {field = "variable", type = "nominal", sort = "-x", axis = {title = "gene"}}
        }
    )(
        DataFrames.stack(df_rec, 1:nvar)
    )

    # plots
    df_pca = DataFrame(z', :auto)
    rename!(df_pca, Symbol.(["z" * string(i) for i in collect(1:nvar)]))
    df_pca[!, :id] = 1:nobs

    @vlplot(:rect, "id:o", "variable:o", color = :value)(DataFrames.stack(df_pca, 1:nvar))

    df_pca[!, :type] = repeat([1, 2]; inner=nobs ÷ 2)
    @vlplot(:point, x = :z1, y = :z2, color = "type:n")(df_pca)
```

<!--  
#### Number of components¶

A direct question arises from this is: How many dimensions do we want to keep in order to represent the latent structure in the data? This is a very central question for all latent factor models, i.e. how many dimensions are needed to represent that data in the latent space. In the case of PCA, there exist a lot of heuristics to make that choice. By using the pPCA model, this can be accomplished very elegantly, with a technique called Automatic Relevance Determination(ARD). Essentially, we are using a specific prior over the factor loadings W that allows us to prune away dimensions in the latent space. The prior is determined by a precision hyperparameter \alpha. Here, smaller values of \alpha correspond to more important components. You can find more details about this in the Bishop book mentioned in the introduction.

```julia
    ppca_ARD = pPCA_ARD(X)
    chain_pccaARD = sample(ppca_ARD, NUTS(), 500)

    StatsPlots.plot(group(chain_pccaARD, :alpha))

 
    # convergence of the chains for the \alpha parameter. This parameter determines the relevance of individual components. We can see that the chains have converged and the posterior of the alpha parameters is centered around much smaller values in two instances. Below, we will use the mean of the small values to select the relevant dimensions - we can clearly see based on the values of \alpha that there should be two dimensions in this example.

    # Extract parameter estimates for plotting - mean of posterior
    w = permutedims(reshape(mean(group(chain_pccaARD, :w))[:, 2], (nvar, nvar)))
    z = permutedims(reshape(mean(group(chain_pccaARD, :z))[:, 2], (nvar, nobs)))'
    α = mean(group(chain_pccaARD, :alpha))[:, 2]
    α


    # We can inspect alpha to see which elements are small, i.e. have a high relevance.

    alpha_indices = sortperm(α)[1:2]
    X = w[alpha_indices, alpha_indices] * z[alpha_indices, :]

    df_rec = DataFrame(X', :auto)
    df_rec[!, :id] = 1:nobs
    @vlplot(:rect, "id:o", "variable:o", color = :value)(DataFrames.stack(df_rec, 1:2))

    df_pre = DataFrame(z', :auto)
    rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:nobs)]))
    df_pre[!, :id] = 1:nobs

    @vlplot(:rect, "id:o", "variable:o", color = :value)(DataFrames.stack(df_pre, 1:nobs))

    df_pre[!, :type] = repeat([1, 2]; inner=nobs ÷ 2)
    df_pre[!, :ard1] = df_pre[:, alpha_indices[1]]
    df_pre[!, :ard2] = df_pre[:, alpha_indices[2]]
    @vlplot(:point, x = :ard1, y = :ard2, color = "type:n")(df_pre)
```


This plot is very similar to the low-dimensional plot above, but choosing the relevant dimensions based on the values of

. When you are in doubt about the number of dimensions to project onto, ARD might provide an answer to that question.


#### Batch effects¶

A second, common aspect apart from the dimensionality of the PCA space, is the issue of confounding factors or batch effects. A batch effect occurs when non-biological factors in an experiment cause changes in the data produced by the experiment. As an example, we will look at Fisher's famous Iris data set.

The data set consists of 50 samples each from three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. RDatasets.jl contains the Iris dataset.

An example for a batch effect in this case might be different scientists using a different measurement method to determine the length and width of the flowers. This can lead to a systematic bias in the measurement unrelated to the actual experimental variable - the species in this case.


```julia

    # First, let's look at the original data using the pPCA model.

    ppca = pPCA(X)

    # Hamiltonian Monte Carlo (HMC) sampler parameters
    ϵ = 0.05
    τ = 10
    chain_ppca2 = sample(ppca, HMC(ϵ, τ), n_samples)

    # Extract parameter estimates for plotting - mean of posterior
    w = permutedims(reshape(mean(group(chain_ppca2, :w))[:, 2], (nvar, nvar)))
    z = permutedims(reshape(mean(group(chain_ppca2, :z))[:, 2], (nvar, nobs)))'
    mu = mean(group(chain_ppca2, :m))[:, 2]

    Xbt = w * z
    # Xbt = w * z .+ mu

    df_rec = DataFrame(Xbt', :auto)
    df_rec[!, :species] = Xdata.Species
    @vlplot(:rect, "species:o", "variable:o", color = :value)(DataFrames.stack(df_rec, 1:nvar))

    df_Xdata = DataFrame(z', :auto)
    rename!(df_Xdata, Symbol.(["z" * string(i) for i in collect(1:nvar)]))
    df_Xdata[!, :sample] = 1:nobs
    df_Xdata[!, :species] = Xdata.Species
    @vlplot(:point, x = :z1, y = :z2, color = "species:n")(df_Xdata)

    # We can see that the setosa species is more clearly separated from the other two species, which overlap considerably.

    # We now simulate a batch effect; imagine the person taking the measurement uses two different rulers and they are slightly off. Again, in practice there are many different reasons for why batch effects occur and it is not always clear what is really at the basis of them, nor can they always be tackled via the experimental setup. So we need methods to deal with them.

    ## Introduce batch effect
    batch = rand(Binomial(1, 0.5), nobs)
    effect = rand(Normal(2.4, 0.6), nobs)
    batch_dat = (X' .+ batch .* effect)'

    ppca_batch = pPCA(batch_dat)
    chain_ppcaBatch = sample(ppca_batch, HMC(ϵ, τ), n_samples)
    describe(chain_ppcaBatch)[1]

    z = permutedims(reshape(mean(group(chain_ppcaBatch, :z))[:, 2], (nvar, nobs)))'
    df_pre = DataFrame(z', :auto)
    rename!(df_pre, Symbol.(["z" * string(i) for i in collect(1:nvar)]))
    df_pre[!, :sample] = 1:nobs
    df_pre[!, :species] = Xdata.id
    df_pre[!, :batch] = batch
    @vlplot(:point, x = :z1, y = :z2, color = "species:n", shape = :batch)(df_pre)

    # The batch effect makes it much harder to distinguish the species. And importantly, if we are not aware of the batches, this might lead us to make wrong conclusions about the data.

    # In order to correct for the batch effect, we need to know about the assignment of measurement to batch. In our example, this means knowing which ruler was used for which measurement, here encoded via the batch variable.

    ppca_residual = pPCA_residual(batch_dat, convert(Vector{Float64}, batch))
    chain_ppcaResidual = sample(ppca_residual, HMC(ϵ, τ), n_samples);

    # This model is described in considerably more detail here.

    z = permutedims(reshape(mean(group(chain_ppcaResidual, :z))[:, 2], (nvar, nobs)))'
    df_post = DataFrame(z', :auto)
    rename!(df_post, Symbol.(["z" * string(i) for i in collect(1:nvar)]))
    df_post[!, :sample] = 1:nobs
    df_post[!, :species] = Xdata.id
    df_post[!, :batch] = batch

    @vlplot(:point, x = :z1, y = :z2, color = "species:n", shape = :batch)(df_post)

    # We can now see, that the data are better separated in the latent space by accounting for the batch effect. It is not perfect, but definitely an improvement over the previous plot.
```

-->
