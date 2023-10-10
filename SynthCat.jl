# from here on actual code to do generate the initial catalogue of objects
# they will need to be perturbed and so on before being properly usable
using PyCall, Statistics, PyPlot, ProgressBars
DBOffline = jld2toDF("/media/data/Data/Catalogues/PS_Selection_AllSources/PS_Filtered_MS_with_meta.jld2");

# load synth data
synth = fits2df("/media/data/Data/Extra/SinteticiSara/Output/photometric_points_colours.fits");
meta = fits2df("/media/data/Data/Extra/SinteticiSara/Output/assigned_parameters.fits");
data = hcat(meta, synth);

# select only QSOs, y_lim is forced by outliers at low z
QSO = @subset(DBOffline, :otypeid .== 1, :ps_y .> 14);
remove_missing!(QSO);

# number of scaled objects that you get out of the script
nObjTarget = 10000;

# import python stuff, might come handy!
mpl = pyimport("matplotlib");
np = pyimport("numpy");
scps = pyimport("scipy.stats");

# generate random numbers using a given distribution (histogram)
function genFromHist(listIn::Array{T,1} where {T<:AbstractFloat}, n::Int; bins::Int=50, bw=0.1)
    function kde(x, x_grid; bandwidth=0.2)
        kde = scps.gaussian_kde(x, bw_method=bandwidth / std(x, corrected=true))
        return kde.evaluate(x_grid)
    end

    x_grid = np.linspace(minimum(listIn), maximum(listIn), 1000)
    kdepdf = kde(listIn, x_grid, bandwidth=bw) # check value

    cdf = np.cumsum(kdepdf)
    cdf = cdf / cdf[end]
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = x_grid[value_bins.+1]
end

scaled_synth = similar(data, 0);

# actually produce data
for zTarget in ProgressBar(0:0.1:5.5)
    subset_data = @subset(data, zTarget - 0.1 .< :z_rand .<= zTarget)

    # Evito problemi nel caso non ci siano dati in quel range di redshift
    if length(subset_data[!, 1]) == 0
        continue
    end

    # We use ymag, that is always there (effectively this is the ycolor, that is also always there and always always zero)
    # For now we also use the actual histogram in bins of 0.1, instead of grouping stuff together
    #  but this will surely break for QSOs at higher redshift. Meh.

    # The number of objects in yMagGenerate sets the number of QSOs per 0.1 of redshift
    # This reproduces the actual distribution of objects in magnitude, so it is not what we want, but it is useful for testing.
    yMagDistribution = @subset(QSO, zTarget - 0.1 .<= :z_spec .< zTarget).ps_y
    yMagGenerate = genFromHist(yMagDistribution, nObjTarget, bins=20)

    for yMagTarget in yMagGenerate
        randId = rand(1:size(subset_data)[1])

        row = subset_data[randId, :]
        ymag = row.PanSTARRS_Y
        ymagScaled = yMagTarget

        delta = ymagScaled - ymag # this is essentially useless, ymag è sempre nullo nei colori
        #  lo lascio per ricordarmi di cosa diavolo sto facendo

        for e in names(row)[5:end]
            row[e] += delta
            #row[e] = round(row[e], digits=4) -> not sure why I was rounding
        end

        push!(scaled_synth, row)
    end
end

scaled_synth;

# now onto introducting upper limits

# now onto generating errors. To do so, we take the function from above, generate the error distribution using GP, and then add errors manually
# TODO: check which colours i need to take for WISE and so on, otherwise there are issues
# Might as well write code to take care of the errors, that I need to do it anyway at some point

function add_iid!(df)
    if "iid" in names(df)
        return
    end
    df_names = names(df)
    df[!, :iid] .= collect(1:size(df)[1])
    pushfirst!(df_names, "iid")
    select!(df, df_names)
end


function preGenErr(df, magNames, errNames, dr3, magNames_dr3, errNames_dr3)
    dfOut = DataFrame(iid=collect(1:length(df.RefRedshift)))
    for (m, e, mdr3, edr3) in zip(magNames, errNames, magNames_dr3, errNames_dr3)
        err = genAllError(df[!, m], mdr3, edr3, 0.1; p=false)
        dfOut[!, e] = err
    end
    return select(dfOut, Not(:iid))
end


function gen_err_function(m, e, str; k=5, s=0.1, delete_hist=true, fix_outliers=false, interp_func=BSplineInterpolation, opts=[3, :ArcLen, :Average], save=false)
    bkp_m, bkp_e = deepcopy(m), deepcopy(e)

    plt.close()
    out = hist2D(both_safe(m, e)..., bins=100)
    if delete_hist
        plt.close()
    end

    mag_pts = Array{Float64,1}()
    err_pts = Array{Float64,1}()

    for i in 1:size(out[1])[1]
        # find max for col
        max_ind = argmax(out[1][i, :])
        push!(mag_pts, out[2][i])
        push!(err_pts, out[3][max_ind])
    end

    # need to make sure that there are no points too far away before splining
    if isa(fix_outliers, Number)
        inds = findall(.!((mag_pts .> fix_outliers) .& (err_pts .< 0.05)))
        mag_pts = mag_pts[inds]
        err_pts = err_pts[inds]
    end

    spl = interp_func(mag_pts, err_pts, opts...)
    m, s = predict_y(spl, mag_pts)

    fix_monotonic(mag_pts, m, fix_outliers - 2)

    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    hist2D(both_safe(bkp_m, bkp_e)..., bins=100)
    plt.scatter(mag_pts, err_pts, s=3)
    plt.plot(mag_pts, m, color="red")

    if save
        plt.savefig("/tmp/$(str).png")
    end

    return spl, mag_pts, err_pts
end


function both_safe(l1, l2)
    inds = findall(isfinite.(l1) .& isfinite.(l2))
    return l1[inds], l2[inds]
end


function fix_monotonic(x, m, th)
    const_val = 0
    inds = findall(x .> th)
    for i in range(1, length(m) - 1)
        if i in inds
            m_i = m[i]
            m_ip1 = m[i+1]
            if m_ip1 > m_i
                const_val = m_i
                continue
            else
                m[i] = const_val
            end
        end
    end
    if m[end] < m[end-1]
        m[end] = m[end-1]
    end
end

# do create error functions
using GaussianProcesses

mag_names = ["ps_g", "ps_r", "ps_i", "ps_z", "ps_y", "cw_w1", "cw_w2", "aw_w3", "aw_w4", "g_g", "g_bp", "g_rp"];
err_names = ["ps_e_g", "ps_e_r", "ps_e_i", "ps_e_z", "ps_e_y", "cw_e_w1", "cw_e_w2", "aw_e_w3", "aw_e_w4", "g_e_g", "g_e_bp", "g_e_rp"];
limit_outlier = [25, 25, 25, 20, 25, 25, 20, 19, 16, 25, 22, 20];

preGenErrDict = Dict{String,Any}();

for (m, e, l) in zip(mag_names, err_names, limit_outlier)
    # this essenially contains all the information about the error function
    # keep in mind, you possibly need a cutoff
    preGenErrDict[m] = gen_err_function(QSO[!, m], QSO[!, e], m, fix_outliers=l, interp_func=GP, opts=[MeanZero(), SE(0.0, 0.0), -1.0])
end

# actual generation of errors
mag_names_synth = ["PanSTARRS_g", "PanSTARRS_r", "PanSTARRS_i", "PanSTARRS_z", "PanSTARRS_Y", "WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4", "Gaia_G", "Gaia_BP", "Gaia_RP"];
for (name_s, name_r) in ProgressBar(zip(mag_names_synth, mag_names))
    scaled_synth[!, "e_"*name_s] .= predict_y(preGenErrDict[name_r][1], scaled_synth[!, name_s])[1]
end

# change the magnitudes so that there is some kind of perturbation and it is not too uniform
# note: here we are making changes in mag! might need to do everything in flux at some point
using Distributions
for name_s in mag_names_synth
    scaled_synth[!, "p_"*name_s] .= rand.(Normal.(scaled_synth[!, name_s], scaled_synth[!, "e_"*name_s]))
end

# recompute errors after having computed the perturbed mags
for (name_s, name_r) in ProgressBar(zip(mag_names_synth, mag_names))
    scaled_synth[!, "p_e_"*name_s] .= predict_y(preGenErrDict[name_r][1], scaled_synth[!, "p_"*name_s])[1]
end

# Upper limits from the Panstarrs catalogue
function newMag(m::Float64, s::Float64)::Float64
    K = 10
    num = (K^2 * s / 2) + ℯ^(-K) * (K * s + m + s) - m - s + K * m
    den = (K + ℯ^(-K) - 1)
    return round(num / den, digits=3)
end

function newMag(m::Float64, s::Float64, N::Float64)::Float64
    return round(m + s * log(1 / (1 - N)), digits=3)
end

# ----------------------------------- #

function probDist(x)
    return x > 0 ? 1 - ℯ^(-x) : 0
end

# ----------------------------------- #

function subLimMag(m, e, name)

    lmList = (
        g_bp=20.96,
        g_g=20.94,
        g_rp=19.63,
        cw_w1=NaN32,
        cw_w2=16.57 + 3.339,
        aw_w3=12.67 + 5.174,
        aw_w4=9.18 + 6.62,
        ps_g=23.71,
        ps_r=23.08,
        ps_i=22.82,
        ps_z=22.82,
        ps_y=NaN32)

    deltaList = (
        g_bp=0.38,
        g_g=0.25,
        g_rp=0.44,
        cw_w1=NaN32,
        cw_w2=0.45,
        aw_w3=0.33,
        aw_w4=0.25,
        ps_g=1.00,
        ps_r=1.15,
        ps_i=0.75,
        ps_z=1.12,
        ps_y=NaN32)

    turnMag = lmList[Symbol(name)]
    if isnan(turnMag)
        return m, 0
    end

    errDist = deltaList[Symbol(name)]
    delta = (m - turnMag) / errDist

    r = rand(Uniform(0, 1))

    if r < probDist(delta)
        return newMag(turnMag, errDist), 1
    else
        return m, 0
    end
end

for (name_s, name_r) in ProgressBar(zip(mag_names_synth, mag_names))
    tmp_val = []
    tmp_map = []

    tmp = subLimMag.(scaled_synth[!, "p_"*name_s], scaled_synth[!, "p_e_"*name_s], name_r)
    for _tmp in tmp
        push!(tmp_val, _tmp[1])
        push!(tmp_map, _tmp[2])
    end
    scaled_synth[!, "ul_"*name_s] = tmp_val
    scaled_synth[findall(tmp_map .== 1), "p_"*name_s] .= NaN
end