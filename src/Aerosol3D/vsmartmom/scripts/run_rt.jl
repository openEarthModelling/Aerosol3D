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
    tau = ds_in["tau"][:, :]         # [wavelength, layer]
    SSA = ds_in["SSA"][:]
    beta = ds_in["beta"][:, :]       # [wavelength, legendre_order]
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
        β = beta[i_wl, :]
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
    # 4. Run RT
    # ------------------------------------------------------------------
    println("\n  Running RT...")
    R, T = rt_run(model)
    println("  RT complete. R shape: ", size(R), ", T shape: ", size(T))

    # ------------------------------------------------------------------
    # 5. Write output NetCDF
    # ------------------------------------------------------------------
    ds_out = NCDataset(output_path, "c")
    defDim(ds_out, "stokes", size(R, 1))
    defDim(ds_out, "vza", n_vza)
    defDim(ds_out, "wavelength", n_wl)
    defDim(ds_out, "layer", n_layers)

    defVar(ds_out, "R", R, ("stokes", "vza", "wavelength"))
    defVar(ds_out, "T", T, ("stokes", "vza", "wavelength"))
    defVar(ds_out, "wavelength_nm", wavelengths, ("wavelength",))
    defVar(ds_out, "wavenumber_cm", wavenumbers, ("wavelength",))
    defVar(ds_out, "vza_deg", vza, ("vza",))
    defVar(ds_out, "vaz_deg", vaz, ("vza",))
    defVar(ds_out, "sza_deg", sza, ())
    defVar(ds_out, "tau_per_layer", tau[:, 1:n_layers], ("wavelength", "layer"))

    ds_out.attrib["source"] = "Aerosol3D-vSmartMOM"
    ds_out.attrib["n_layers"] = n_layers
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
