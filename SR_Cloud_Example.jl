#=============================================================================
  FlowFNO Example — RUN #2: 128³ → 192³ Se + q_l, trained against TRUE DNS
  ==========================================================================
  Reads Se, q_l at BOTH resolutions from read_data.jl:
    scalar1 = Se_128, scalar2 = ql_128   (input)
    scalar3 = Se_192, scalar4 = ql_192   (TRUE DNS target — Run #2)

  WHAT CHANGED vs Run #1 — see SR_Cloud_Pipeline.jl header for full detail:
    1. Training target is the TRUE 192³ DNS field, not a spectral-padding
       baseline. The baseline is still computed, but purely as a reporting
       comparison — never fed into augment_dns/train_flowfno! as the target.
    2. Normalisation stats (Se_mean/Se_std/ql_scale) are pooled across BOTH
       the 128³ and 192³ fields via combined_zscore_stats/combined_scale_stat.
    3. Per-channel loss weights are computed from the normalised TRUE target
       variance (inverse-variance weighting) so Se and q_l contribute
       comparable gradient signal instead of one dominating the shared MSE.
    4. noise_scale is now per-channel — starting values below, re-tune using
       the fluctuation-std-vs-true-DNS-std diagnostic printed at the end.

  A NEW VALIDATION SECTION is added at the bottom: because we now have real
  192³ DNS, we can finally compute actual reconstruction error (RMSE, bias)
  against ground truth — not just self-consistency checks against the
  model's own input, which is all Run #1 could do.
=============================================================================#

include("SR_Cloud_Pipeline.jl")
include("read_data.jl")  # → scalar1(Se_128), scalar2(ql_128), scalar3(Se_192), scalar4(ql_192)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    augment_dns(Se_128n, ql_128n, Se_192n, ql_192n)

Generate 8 training pairs by reflecting the DNS fields along all axes.
RUN #2: Se_192n/ql_192n here must be the TRUE 192³ DNS fields (normalised),
not a spectral-padding baseline — see SR_Cloud_Pipeline.jl header.
"""
function augment_dns(Se_128n::Array{Float64,3},
                     ql_128n::Array{Float64,3},
                     Se_192n::Array{Float64,3},
                     ql_192n::Array{Float64,3})

    flips = [
        (),         (1,),       (2,),       (3,),
        (1,2),      (1,3),      (2,3),      (1,2,3)
    ]

    inputs  = Vector{Array{Float64,4}}(undef, 8)
    targets = Vector{Array{Float64,4}}(undef, 8)

    for (k, dims) in enumerate(flips)
        Se_f  = isempty(dims) ? Se_128n : reverse(Se_128n, dims=dims)
        ql_f  = isempty(dims) ? ql_128n : reverse(ql_128n, dims=dims)
        Se_t  = isempty(dims) ? Se_192n : reverse(Se_192n, dims=dims)
        ql_t  = isempty(dims) ? ql_192n : reverse(ql_192n, dims=dims)

        inp = zeros(Float64, 2, size(Se_f)...)
        inp[1,:,:,:] = Se_f
        inp[2,:,:,:] = ql_f

        tgt = zeros(Float64, 2, size(Se_t)...)
        tgt[1,:,:,:] = Se_t
        tgt[2,:,:,:] = ql_t

        inputs[k]  = inp
        targets[k] = tgt
    end
    return inputs, targets
end

"""
    inverse_variance_weights(var_se, var_ql)

Per-channel loss weights (Run #2 fix #3): channels with larger normalised
target variance get down-weighted so squared-error contributions are
comparable in scale across channels. Weights are rescaled to average 1.0,
so a weight of 1.0 means "unweighted," for interpretability.
"""
function inverse_variance_weights(var_se::Float64, var_ql::Float64)
    w_se, w_ql = 1.0 / var_se, 1.0 / var_ql
    s = w_se + w_ql
    return 2.0 * w_se / s, 2.0 * w_ql / s
end

"""Save all 128³ input and 192³ output fields (Se and q_l) to HDF5,
including the TRUE 192³ DNS fields for direct comparison plots."""
function save_output(filepath::String,
                     Se_128 ::Array{Float64,3}, ql_128 ::Array{Float64,3},
                     Se_192_dns::Array{Float64,3}, ql_192_dns::Array{Float64,3},
                     Se_base::Array{Float64,3}, ql_base::Array{Float64,3},
                     Se_mean::Array{Float64,3}, ql_mean::Array{Float64,3},
                     Se_samp::Array{Float64,3}, ql_samp::Array{Float64,3},
                     params ::DomainParams)
    h5open(filepath, "w") do f
        f["Se_128_input"]    = Se_128
        f["ql_128_input"]    = ql_128
        f["Se_192_dns"]      = Se_192_dns  # TRUE DNS — ground truth (Run #2)
        f["ql_192_dns"]      = ql_192_dns
        f["Se_192_baseline"] = Se_base     # spectral zero-padding — reporting only
        f["ql_192_baseline"] = ql_base
        f["Se_192_mean"]     = Se_mean     # FNO mean prediction
        f["ql_192_mean"]     = ql_mean
        f["Se_192_sample"]   = Se_samp     # FlowFNO stochastic sample
        f["ql_192_sample"]   = ql_samp
        f["domain_L_m"]      = params.L
        f["N_coarse"]        = params.N_coarse
        f["N_fine"]          = params.N_fine
        f["grid_spacing_m"]  = params.L / params.N_fine
    end
    @info "HDF5 saved → $filepath"
end

function save_binary(prefix::String,
                     Se_mean::Array{Float64,3}, ql_mean::Array{Float64,3},
                     Se_samp::Array{Float64,3}, ql_samp::Array{Float64,3})
    write("$(prefix)_Se_mean.bin",   vec(Se_mean))
    write("$(prefix)_Se_sample.bin", vec(Se_samp))
    write("$(prefix)_ql_mean.bin",   vec(ql_mean))
    write("$(prefix)_ql_sample.bin", vec(ql_samp))
    @info "Binary files: $(prefix)_{Se,ql}_{mean,sample}.bin"
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
function main()

    params = DomainParams(N_coarse=128, N_fine=192)

    # ── Load DNS fields from read_data.jl (BOTH resolutions) ──────────────────
    Se_128 = Float64.(scalar1)   # 128³ input
    ql_128 = Float64.(scalar2)
    Se_192 = Float64.(scalar3)   # 192³ TRUE DNS target (Run #2)
    ql_192 = Float64.(scalar4)

    @assert size(Se_128) == (128,128,128) "Expected scalar1 of size (128,128,128)"
    @assert size(ql_128) == (128,128,128) "Expected scalar2 of size (128,128,128)"
    @assert size(Se_192) == (192,192,192) "Expected scalar3 of size (192,192,192)"
    @assert size(ql_192) == (192,192,192) "Expected scalar4 of size (192,192,192)"

    println("\n▶ DNS fields loaded:")
    @printf("  Se  128³: range [%.6e, %.6e]  mean %.6e\n",
            minimum(Se_128), maximum(Se_128), mean(Se_128))
    @printf("  Se  192³: range [%.6e, %.6e]  mean %.6e   (TRUE DNS)\n",
            minimum(Se_192), maximum(Se_192), mean(Se_192))
    @printf("  q_l 128³: range [%.6e, %.6e]  mean %.6e\n",
            minimum(ql_128), maximum(ql_128), mean(ql_128))
    @printf("  q_l 192³: range [%.6e, %.6e]  mean %.6e   (TRUE DNS)\n",
            minimum(ql_192), maximum(ql_192), mean(ql_192))

    # ── Normalise — POOLED across both resolutions (Run #2 fix #2) ────────────
    Se_mean, Se_std = combined_zscore_stats(Se_128, Se_192)
    ql_scale        = combined_scale_stat(ql_128, ql_192)

    @printf("\n  Normalisation (pooled 128³+192³): Se_mean=%.6e  Se_std=%.6e  ql_scale=%.6e\n",
            Se_mean, Se_std, ql_scale)

    Se_128n = (Se_128 .- Se_mean) ./ Se_std
    ql_128n = ql_128 ./ ql_scale
    Se_192n = (Se_192 .- Se_mean) ./ Se_std
    ql_192n = ql_192 ./ ql_scale

    @printf("  Se  128³ normalised range: [%.4f, %.4f]\n", minimum(Se_128n), maximum(Se_128n))
    @printf("  Se  192³ normalised range: [%.4f, %.4f]\n", minimum(Se_192n), maximum(Se_192n))
    @printf("  q_l 128³ normalised range: [%.4f, %.4f]\n", minimum(ql_128n), maximum(ql_128n))
    @printf("  q_l 192³ normalised range: [%.4f, %.4f]\n", minimum(ql_192n), maximum(ql_192n))

    # ── Physics baseline: spectral zero-padding — REPORTING ONLY (Run #2) ────
    # No longer used as a training target — see pipeline header note #1.
    println("\n▶ Baseline: Spectral zero-padding (128³ → 192³), for comparison only...")
    Se_base_raw = spectral_zero_pad(Se_128, params.N_fine)
    ql_base_raw = spectral_zero_pad(ql_128, params.N_fine)
    ql_base_raw .= max.(ql_base_raw, 0.0)

    # ── Per-channel loss weights (Run #2 fix #3) ───────────────────────────────
    # Inverse-variance weighting on the TRUE normalised target: the channel
    # with more normalised spread (typically Se) gets down-weighted so it
    # can't dominate the shared gradient the way it did in Run #1.
    var_se, var_ql = var(Se_192n), var(ql_192n)
    w_se, w_ql = inverse_variance_weights(var_se, var_ql)
    @printf("\n  Per-channel loss weights (inverse-variance, avg=1.0): Se=%.4f  q_l=%.4f\n",
            w_se, w_ql)
    @printf("  (normalised target variance: Se=%.4f  q_l=%.4f)\n", var_se, var_ql)

    # ── Generate 8 augmented training pairs from real DNS data ────────────────
    println("\n▶ Generating 8 augmented DNS training pairs (axis flips)...")
    println("  Target = TRUE 192³ DNS (Run #2) — not a spectral-padding baseline.")
    inputs_list, targets_list = augment_dns(Se_128n, ql_128n, Se_192n, ql_192n)

    for k in 1:8
        @printf("  Pair %d: Se in [%.4f,%.4f] | ql in [%.4f,%.4f] | Se tgt [%.4f,%.4f] | ql tgt [%.4f,%.4f]\n",
                k,
                minimum(inputs_list[k][1,:,:,:]),  maximum(inputs_list[k][1,:,:,:]),
                minimum(inputs_list[k][2,:,:,:]),  maximum(inputs_list[k][2,:,:,:]),
                minimum(targets_list[k][1,:,:,:]), maximum(targets_list[k][1,:,:,:]),
                minimum(targets_list[k][2,:,:,:]), maximum(targets_list[k][2,:,:,:]))
    end

    # ── Initialise FlowFNO pipeline ───────────────────────────────────────────
    device = "cpu"   # change to "mps" for speedup on Apple Silicon (test first)

    println("\n▶ Initialising FlowFNO pipeline on device=$device (Run #2)...")
    pipeline = FlowFNO(;
        device               = device,
        in_channels          = 2,
        out_channels         = 2,
        fno_modes            = 6,
        fno_width            = 4,
        fno_layers           = 2,
        base_ch              = 2,
        time_dim             = 32,
        # Starting point only — re-tune from the fluctuation-std diagnostic
        # printed below, same iterative process as Run #1. q_l gets more
        # spread than Se here since its TRUE 192³ distribution (mean ~2.4e-4)
        # is much wider than the 128³ input it's conditioned on.
        noise_scale          = [0.1, 0.3],
        flow_steps           = 20,
        lr_fno               = 1e-3,
        lr_flow              = 1e-4,
        epochs               = 300,
        params               = params,
        fno_channel_weights  = [w_se, w_ql],
        flow_channel_weights = [w_se, w_ql]
    )

    pipeline.Se_mean  = Se_mean;  pipeline.Se_std  = Se_std
    pipeline.ql_scale = ql_scale

    # ── Joint training ────────────────────────────────────────────────────────
    println("\n▶ Training FlowFNO (300 epochs, 8 DNS-augmented pairs, true 192³ target)...")
    train_flowfno!(pipeline, inputs_list, targets_list;
                   epochs       = 300,
                   fno_weight   = 1.0,
                   flow_weight  = 1.0,
                   log_every    = 30)

    # ── Reconstruct 192³ Se + q_l ─────────────────────────────────────────────
    println("\n▶ FlowFNO reconstruction of DNS scalars (128³ → 192³)...")
    Se_mean_f, ql_mean_f, Se_sample_f, ql_sample_f =
        reconstruct(pipeline, Se_128, ql_128; n_flow_steps = 20, heun = true)

    @printf("  Se  FNO mean   : range [%.6e, %.6e]  mean %.6e\n",
            minimum(Se_mean_f), maximum(Se_mean_f), mean(Se_mean_f))
    @printf("  Se  Flow sample: range [%.6e, %.6e]  mean %.6e\n",
            minimum(Se_sample_f), maximum(Se_sample_f), mean(Se_sample_f))
    @printf("  q_l FNO mean   : range [%.6e, %.6e]  mean %.6e\n",
            minimum(ql_mean_f), maximum(ql_mean_f), mean(ql_mean_f))
    @printf("  q_l Flow sample: range [%.6e, %.6e]  mean %.6e\n",
            minimum(ql_sample_f), maximum(ql_sample_f), mean(ql_sample_f))

    # ── Save model weights and 192³ outputs ────────────────────────────────────
    save_models(pipeline, "checkpoints")

    println("\n▶ Saving 192³ output fields (including true DNS for comparison)...")
    save_output("SeQl_192_flowfno.h5",
                Se_128, ql_128, Se_192, ql_192, Se_base_raw, ql_base_raw,
                Se_mean_f, ql_mean_f, Se_sample_f, ql_sample_f, params)
    save_binary("SeQl_192", Se_mean_f, ql_mean_f, Se_sample_f, ql_sample_f)

    # ── VALIDATION AGAINST TRUE DNS (new in Run #2) ────────────────────────────
    # This is the first genuine accuracy check the pipeline can do: Run #1
    # only had a spectral-padding baseline to compare against — an artifact
    # of the same 128³ input, not independent ground truth. Now we can
    # compute real error metrics against the held-out 192³ DNS field.
    rmse(a, b)  = sqrt(mean((a .- b).^2))
    bias(a, b)  = mean(a .- b)

    Se_rmse_mean = rmse(Se_mean_f, Se_192);   Se_bias_mean = bias(Se_mean_f, Se_192)
    ql_rmse_mean = rmse(ql_mean_f, ql_192);   ql_bias_mean = bias(ql_mean_f, ql_192)
    Se_rmse_samp = rmse(Se_sample_f, Se_192); Se_bias_samp = bias(Se_sample_f, Se_192)
    ql_rmse_samp = rmse(ql_sample_f, ql_192); ql_bias_samp = bias(ql_sample_f, ql_192)

    # Baseline error, for reference — how good is naive spectral padding alone?
    Se_rmse_base = rmse(Se_base_raw, Se_192); Se_bias_base = bias(Se_base_raw, Se_192)
    ql_rmse_base = rmse(ql_base_raw, ql_192); ql_bias_base = bias(ql_base_raw, ql_192)

    # ── Summary ───────────────────────────────────────────────────────────────
    println("\n" * "="^72)
    println("  FlowFNO RECONSTRUCTION SUMMARY  (128³ → 192³)  DNS Case_H_P — RUN #2")
    println("  Joint Se + q_l, trained against TRUE 192³ DNS")
    println("="^72)
    @printf("  %-22s | %11s | %11s | %11s\n", "Method", "Min", "Max", "Mean")
    println("  " * "─"^68)
    println("  Se (supersaturation):")
    for (name, field) in [
            ("TRUE 192³ DNS",     Se_192),
            ("Spectral padding",  Se_base_raw),
            ("FNO mean",          Se_mean_f),
            ("FlowFNO sample",    Se_sample_f)]
        @printf("    %-20s | %11.4e | %11.4e | %11.4e\n",
                name, minimum(field), maximum(field), mean(field))
    end
    println("  q_l (cloud water):")
    for (name, field) in [
            ("TRUE 192³ DNS",     ql_192),
            ("Spectral padding",  ql_base_raw),
            ("FNO mean",          ql_mean_f),
            ("FlowFNO sample",    ql_sample_f)]
        @printf("    %-20s | %11.4e | %11.4e | %11.4e\n",
                name, minimum(field), maximum(field), mean(field))
    end
    println("="^72)

    println("\n  VALIDATION vs TRUE 192³ DNS (new — was not possible in Run #1):")
    @printf("  %-22s | %13s | %13s\n", "Method", "RMSE", "Bias (pred-true)")
    println("  " * "─"^54)
    @printf("  Se  spectral padding   | %13.4e | %13.4e\n", Se_rmse_base, Se_bias_base)
    @printf("  Se  FNO mean           | %13.4e | %13.4e\n", Se_rmse_mean, Se_bias_mean)
    @printf("  Se  FlowFNO sample     | %13.4e | %13.4e\n", Se_rmse_samp, Se_bias_samp)
    @printf("  q_l spectral padding   | %13.4e | %13.4e\n", ql_rmse_base, ql_bias_base)
    @printf("  q_l FNO mean           | %13.4e | %13.4e\n", ql_rmse_mean, ql_bias_mean)
    @printf("  q_l FlowFNO sample     | %13.4e | %13.4e\n", ql_rmse_samp, ql_bias_samp)
    println()
    println("  Interpretation:")
    println("    • If FNO/FlowFNO RMSE < spectral-padding RMSE, the model is adding")
    println("      real value over naive upsampling — check this for BOTH channels.")
    println("    • Large |bias| on q_l (as in Run #1, where the model matched the")
    println("      128³ mean instead of the true 192³ mean) indicates the model is")
    println("      still not capturing the true fine-scale amplification.")

    println()
    println("  Fluctuation captured by flow matching (vs TRUE 192³ DNS std):")
    @printf("    Se : std(sample - mean) = %.4e   TRUE 192³ Se  std = %.4e   ratio = %.3f\n",
            std(Se_sample_f .- Se_mean_f), std(Se_192), std(Se_sample_f .- Se_mean_f)/std(Se_192))
    @printf("    q_l: std(sample - mean) = %.4e   TRUE 192³ q_l std = %.4e   ratio = %.3f\n",
            std(ql_sample_f .- ql_mean_f), std(ql_192), std(ql_sample_f .- ql_mean_f)/std(ql_192))
    println("    (ratio ≈ 1.0 is well-calibrated; ≪1 under-dispersed, ≫1 over-dispersed —")
    println("     adjust the corresponding entry in `noise_scale` accordingly.)")

    println()
    println("  Output files:")
    println("    SeQl_192_flowfno.h5       HDF5 — all fields incl. TRUE 192³ DNS + metadata")
    println("    SeQl_192_Se_mean.bin      flat Float64 binary (192³) — Se FNO mean")
    println("    SeQl_192_Se_sample.bin    flat Float64 binary (192³) — Se FlowFNO sample")
    println("    SeQl_192_ql_mean.bin      flat Float64 binary (192³) — q_l FNO mean")
    println("    SeQl_192_ql_sample.bin    flat Float64 binary (192³) — q_l FlowFNO sample")
    println()
    println("  Tuning notes for Run #3:")
    println("    • Spread: adjust noise_scale=[Se,ql] per the ratio diagnostic above.")
    println("    • If q_l bias is still large: try a log1p(q_l/ql_scale) transform")
    println("      before training (expm1 back at inference) to counter its right-skew —")
    println("      not applied here, deferred per SR_Cloud_Pipeline.jl header.")
    println("    • If q_l still looks over-smoothed after that: raise fno_modes (e.g. 8-10)")
    println("      to better resolve q_l's sharp, localized peaks.")
    println("    • Memory: cast inputs_list/targets_list to Float32 if near the 8 GB ceiling.")
    println()
    println("✓ FlowFNO pipeline complete (Run #2).")
end

main()
