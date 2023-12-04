# PCA Latent Bayesian Rotation Invariant Householder form

## set up environment and test data

```julia
    # load helper files
    project_directory = joinpath( homedir(), "bio", "pca"  )
    
    include( joinpath( project_directory, "startup.jl" ))     
    include( joinpath( project_directory, "pca_functions.jl" ))     
 
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


## PCA, Bayesian Householder transform to remove Rotational symmetry


#### Rationale: 
Householder transform: removes the rotational symmetry from the posterior, improving (and speeding up inference). This is done by first using SVD of $W$ (assuming it is positive definite):

$$\boldsymbol{W} \boldsymbol{W}^T = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T (\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^T)^T = \boldsymbol{U} \boldsymbol{\Sigma}^2 \boldsymbol{U}^T$$ 
where $U$ and $V$ are orthogonal and $\Sigma$ is a diagonal with singular values (eigenvalues of $WW^T$). $V$ is the rotation symmetry to be removed by setting it to I (as is also the case in Clasical PCA). The $U$ is a member of the Stiefel manifold ($U^T U =I$) and thus:

$$p(\boldsymbol{Y} | \boldsymbol{U}, \boldsymbol{\Sigma}) = \prod_{i=1}^{m} \text{N} (\boldsymbol{Y}_{i,.}|0, \boldsymbol{U} \boldsymbol{\Sigma}^2 \boldsymbol{U}^T + \sigma^2 \boldsymbol{I})$$
where $U$ is uniformly distributed on the Stiefel manifold. The Householder transform reflects a vector such that all coordinates disappear except one (QR decomposition). So, for PCA, we have:

$$p(\boldsymbol{W} | \boldsymbol{Y}) = \frac{p(\boldsymbol{Y} | \boldsymbol{W}) \enspace  p(\boldsymbol{W}) }{ p(\boldsymbol{Y} )}$$


```julia

    # include( joinpath( project_directory, "pca_functions.jl" ))     

    Random.seed!(1);
    nq = nvar

    U, sigma_pca, C, PC = pca_standard(X')
    v = eigenvector_to_householder(U, nq)  

    # param sequence = sigma_noise, sigma(nq), v, r=norm(v)~ 1.0 (scaled)
    init_params = [0.1; sigma_pca[1:nq]; v; 1.0 ]

    Mph = PCA_BH_model(X, nq )  # all dims == default form
    chain = sample(Mph, NUTS(), n_samples; init_params=init_params);
    showall(chain)

    scores = PCA_BH_extract( chain, X, nq, return_object="scores" )
    scores_mean = PCA_BH_extract( chain, X, nq, return_object="scores_mean" )
    scores_mean[!, :id] = Xdata.id
    @vlplot(:point, x = :pc1, y = :pc2, color = "id:n")(scores_mean)


    ## Create data projections for each step of chain
  
    sigma = PCA_BH_extract( chain, X, nq, return_object="sigma" )  # posteriors
    
    scores = PCA_BH_extract( chain, X, nq, return_object="scores" )  # posteriors

    plt = plot(
        [100, 200, 300];
        xlim=(-6., 6.),
        ylim=(-6., 6.),
        group=["Setosa", "Versicolor", "Virginica"],
        markercolor=["red", "blue", "black"],
        title="Visualization",
        seriestype=:scatter,
    )
    setosa = 1:50
    versicolor = 51:100
    virginica = 101:150
    scatter!(
        plt,
        scores[1, setosa, :],
        scores[2, setosa, :];
        color="red",
        seriesalpha=0.01,
        label="",
    )
    scatter!(
        plt,
        scores[1, versicolor, :],
        scores[2, versicolor, :];
        color="blue",
        seriesalpha=0.01,
        label="",
    )
    scatter!(
        plt,
        scores[1, virginica, :],
        scores[2, virginica, :];
        color="black",
        seriesalpha=0.01,
        label="",
    )
 

``` 


 
#### References

Rajbir S. Nirwan and Nils Bertschinger, 2019. Proceedings of the 36 th International Conference on Machine Learning, Long Beach, California, PMLR 97, 


- https://github.com/rsnirwan/HouseholderBPCA/blob/master/talk.pdf

- Christopher M. Bishop, Pattern Recognition and Machine Learning, 2006.

- https://turing.ml/v0.21/tutorials/11-probabilistic-pca

- https://proceedings.mlr.press/v97/nirwan19a/nirwan19a.pdf

- https://math.stackexchange.com/questions/4258281/understanding-householder-transformation-geometrically

- https://proceedings.mlr.press/v97/nirwan19a.html

 