# =============================================================================
# run_rt.jl — Julia entrypoint for Aerosol3D → vSmartMOM RT
# =============================================================================
# Usage:
#   julia --project=<vSmartMOM-project> run_rt.jl <input.nc> <output.nc>
#
# Reads bulk aerosol optics + vertical profile from input NetCDF,
# injects into vSmartMOM model, runs RT, writes R and T to output NetCDF.
# =============================================================================

using NCDatasets
using Statistics
using vSmartMOM
using vSmartMOM.Scattering

function find_nearest_band(model, wavelength_nm::Float64)
    """Find the vSmartMOM band index closest to the given wavelength."""
    target_wavenumber = 1e7 / wavelength_nm
    min_diff = Inf
    best_band = 1
    for (band_idx, band) in enumerate(model.params.spec_bands)
        band_vals = collect(band)
        band_center = median(band_vals)
        diff = abs(band_center - target_wavenumber)
        if diff < min_diff
            min_diff = diff
            best_band = band_idx
        end
    end
    return best_band
end

function run_rt(input_path::String, output_path::String)
    # ------------------------------------------------------------------
    # 1. Read input NetCDF
    # ------------------------------------------------------------------
    ds_in = NCDataset(input_path, "r")
    wavelengths = ds_in["wavelength_nm"][:]
    wavenumbers = ds_in["wavenumber_cm"][:]
    # NCDatasets returns arrays in Julia column-major order,
    # which transposes C-order NetCDF arrays
    tau = permutedims(ds_in["tau"][:, :], (2, 1))   # [wavelength, layer]
    SSA = ds_in["SSA"][:]
    beta = permutedims(ds_in["beta"][:, :], (2, 1)) # [wavelength, legendre_order]
    sza = ds_in["sza"][]
    vza = ds_in["vza"][:]
    vaz = ds_in["vaz"][:]
    n_legendre = ds_in.attrib["n_legendre"]
    close(ds_in)

    n_wl = length(wavelengths)
    n_vza = length(vza)

    # ------------------------------------------------------------------
    # 2. Build dummy model (for layer structure, angles, quadrature)
    # ------------------------------------------------------------------
    params = vSmartMOM.default_parameters()
    params.absorption_params = nothing
    params.architecture = vSmartMOM.Architectures.CPU()
    params.sza = sza
    params.vza = vza
    params.vaz = vaz

    # Create one spec band per wavelength (small range around each)
    params.spec_bands = [
        range(1e7 / (wl + 1.0), stop=1e7 / (wl - 1.0), length=3)
        for wl in wavelengths
    ]

    model = model_from_parameters(params)
    n_layers = length(model.profile.p_full)

    # Replicate BRDF for each band (default_parameters creates only 1)
    n_bands = length(model.params.spec_bands)
    while length(model.params.brdf) < n_bands
        push!(model.params.brdf, deepcopy(model.params.brdf[1]))
    end

    println("=== Aerosol3D → vSmartMOM RT ===")
    println("  Wavelengths: ", round.(wavelengths, digits=1), " nm")
    println("  Layers: ", n_layers)
    println("  Angles: SZA=", sza, "°, VZA=", vza)

    # ------------------------------------------------------------------
    # 3. Inject per-band aerosol optics
    # ------------------------------------------------------------------
    for i_wl in 1:n_wl
        band = find_nearest_band(model, wavelengths[i_wl])

        # Overwrite τ_aer (per-layer)
        n_tau_layers = min(n_layers, size(tau, 2))
        model.τ_aer[band][1, 1:n_tau_layers] .= tau[i_wl, 1:n_tau_layers]

        # Overwrite aerosol optics (uniform across layers)
        β_raw = beta[i_wl, :]
        β = Float64.(replace(β_raw, missing => 0.0))
        greek = GreekCoefs(
            zeros(n_legendre),   # α
            β,                   # β
            zeros(n_legendre),   # γ
            zeros(n_legendre),   # δ
            zeros(n_legendre),   # ε
            zeros(n_legendre)    # ζ
        )

        model.aerosol_optics[band][1] = AerosolOptics(
            greek_coefs = greek,
            ω̃ = SSA[i_wl],
            k = 1.0,
            fᵗ = 0.0,
        )

        println("  Band ", band, " @ ", round(wavelengths[i_wl], digits=1),
                "nm: τ_total=", round(sum(tau[i_wl, :]), digits=4),
                ", SSA=", round(SSA[i_wl], digits=4))
    end

    # ------------------------------------------------------------------
    # 4. Run RT per band
    # ------------------------------------------------------------------
    println("\n  Running RT...")
    R_all = []
    T_all = []
    for i_band in 1:n_wl
        R_b, T_b = rt_run(model, i_band = i_band)
        println("  Band ", i_band, " R shape: ", size(R_b), ", T shape: ", size(T_b))
        push!(R_all, Array(R_b))
        push!(T_all, Array(T_b))
    end

    # vSmartMOM returns (vza, stokes, n_internal) per band;
    # take the center spectral point and concatenate
    R = cat([R_all[i][:, :, 2] for i in 1:n_wl]..., dims = 3)  # (vza, stokes, n_wl)
    T = cat([T_all[i][:, :, 2] for i in 1:n_wl]..., dims = 3)  # (vza, stokes, n_wl)
    R = permutedims(R, (2, 1, 3))  # (stokes, vza, wavelength)
    T = permutedims(T, (2, 1, 3))  # (stokes, vza, wavelength)
    println("  Combined R shape: ", size(R), ", T shape: ", size(T))

    n_stokes = size(R, 1)

    # ------------------------------------------------------------------
    # 5. Write output NetCDF
    # ------------------------------------------------------------------
    ds_out = NCDataset(output_path, "c")
    defDim(ds_out, "stokes", n_stokes)
    defDim(ds_out, "vza", n_vza)
    defDim(ds_out, "wavelength", n_wl)

    defVar(ds_out, "R", R, ("stokes", "vza", "wavelength"))
    defVar(ds_out, "T", T, ("stokes", "vza", "wavelength"))
    defVar(ds_out, "wavelength", wavelengths, ("wavelength",))
    defVar(ds_out, "wavenumber", wavenumbers, ("wavelength",))
    defVar(ds_out, "vza", vza, ("vza",))
    defVar(ds_out, "vaz", vaz, ("vza",))
    ds_out.attrib["sza"] = sza
    # Clean tau (may contain Missing from NetCDF fill values)
    tau_clean = Float64.(replace(tau, missing => 0.0))

    # Only output the input layers (not vSmartMOM's internal layers)
    n_tau_layers = size(tau_clean, 2)
    defDim(ds_out, "input_layer", n_tau_layers)
    defVar(ds_out, "tau_per_layer", tau_clean, ("wavelength", "input_layer"))

    ds_out.attrib["source"] = "Aerosol3D-vSmartMOM"
    ds_out.attrib["n_layers"] = n_layers
    ds_out.attrib["n_input_layers"] = n_tau_layers
    ds_out.attrib["n_legendre"] = n_legendre

    close(ds_out)
    println("  Output written to: ", output_path)
end

# Entry point
if length(ARGS) != 2
    println("Usage: julia --project=<project> run_rt.jl <input.nc> <output.nc>")
    exit(1)
end

run_rt(ARGS[1], ARGS[2])
