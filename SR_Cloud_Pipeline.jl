#=============================================================================
  FlowFNO Pipeline — RUN #2: trains against TRUE 192³ DNS
  ==========================================================
  Joint FNO + flow-matching training for SUPERSATURATION (Se) + CLOUD WATER
  (q_l) super-resolution, predicted jointly as a 2-channel field.

  WHAT CHANGED vs Run #1 (based on the 128³/192³ DNS PDF comparison)
  --------------------------------------------------------------------
  Run #1 trained against `spectral_zero_pad(field_128)` as a stand-in target
  — a deterministic upsampling of the SAME 128³ field, containing no genuine
  sub-grid structure. The resulting PDF comparison showed the model's q_l
  output was statistically indistinguishable from the 128³ input (same mean,
  ~1.0e-4), while the true 192³ DNS mean sits at ~2.4e-4 — the model wasn't
  super-resolving q_l at all, just passing it through. Se, whose shape barely
  changes between resolutions, was already fine. Three fixes follow directly
  from that diagnosis:

    1. TRUE TARGET. Training now uses the real 192³ DNS fields
       (`read_data.jl`'s scalar3=Se_192, scalar4=ql_192), not a spectral
       zero-padding of the 128³ input. `spectral_zero_pad` is retained only
       as a reporting baseline for comparison, never as a training target.

    2. COMBINED-RESOLUTION NORMALISATION. `Se_mean/Se_std/ql_scale` are now
       computed from BOTH resolutions pooled together (see
       `combined_zscore_stats` / `combined_scale_stat` below), not just the
       128³ field. Run #1's `ql_scale` (from the 128³ max) undershot the true
       192³ max by >3×, which would have pushed normalised training targets
       far outside a well-conditioned range.

    3. PER-CHANNEL LOSS WEIGHTING. q_l is sparse/low-variance almost
       everywhere; Se is high-variance almost everywhere. A single combined
       MSE over both channels let Se's larger error dominate the shared
       gradient, letting q_l get away with a near-trivial near-mean
       prediction (exactly what Run #1's PDF showed). The reconstruction
       loss and the flow-matching loss are now computed PER CHANNEL and
       combined with separate weights, so neither channel can silently
       out-compete the other for gradient signal.

    Also included: FlowSchedule's `noise_scale` is now a per-channel vector
    (previously one scalar for both channels), since Se and q_l need
    different amounts of stochastic spread to match their (very different)
    fluctuation scales — this is the direct lever for fixing the
    under/over-dispersion asymmetry Run #1 showed in its std-ratio
    diagnostics.

  Everything else — shared multi-channel FNO backbone, the stochastic-
  interpolant flow schedule, the few-step Heun ODE sampler starting from the
  FNO mean (not pure noise), and the single joint backward() pass — is
  unchanged in spirit from Run #1.

  Deferred (not applied here — see comments in SR_Cloud_Example.jl):
    • log1p(q_l) transform to counter its right-skew — worth trying in
      Run #3 if per-channel weighting alone doesn't fully fix the q_l bias.
    • Raising `modes` (Fourier modes) to better resolve q_l's sharp,
      localized peaks — a second-order fix, try only if capacity (not
      target/normalisation/loss-balance) turns out to still be the
      bottleneck after this run.
=============================================================================#

ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"

using PythonCall
using FFTW
using HDF5
using Statistics
using LinearAlgebra
using Printf
using Random

# Multi-threading
FFTW.set_num_threads(Threads.nthreads())
@info "Julia threads: $(Threads.nthreads()) | FFTW threads: $(Threads.nthreads())"

# Persistent Python globals
const torch = PythonCall.pynew()
const nn    = PythonCall.pynew()
const optim = PythonCall.pynew()
const np    = PythonCall.pynew()
const F_py  = PythonCall.pynew()

# Persistent Python namespace for exec/eval
const _NS = pydict()
py_run(code::String)   = pyexec(code, _NS)
py_get(name::String)   = _NS[name]

# ─────────────────────────────────────────────────────────────────────────────
# Domain parameters
# ─────────────────────────────────────────────────────────────────────────────
Base.@kwdef struct DomainParams
    L::Float64          = 0.512
    N_coarse::Int       = 128
    N_fine::Int         = 192
    k_min::Float64      = 2π / 0.512
end

# ─────────────────────────────────────────────────────────────────────────────
# Python initialisation
# ─────────────────────────────────────────────────────────────────────────────
function init_python!()
    PythonCall.pycopy!(torch, pyimport("torch"))
    PythonCall.pycopy!(nn,    pyimport("torch.nn"))
    PythonCall.pycopy!(optim, pyimport("torch.optim"))
    PythonCall.pycopy!(np,    pyimport("numpy"))
    PythonCall.pycopy!(F_py,  pyimport("torch.nn.functional"))

    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore",
        message="An output with one or more elements was resized",
        category=pybuiltins.UserWarning)

    n = Threads.nthreads()
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)
    @info "PyTorch: CUDA=$(pyconvert(Bool,torch.cuda.is_available())), " *
          "MPS=$(pyconvert(Bool,torch.backends.mps.is_available())), " *
          "CPU threads=$n"
end

# ─────────────────────────────────────────────────────────────────────────────
# Julia ↔ PyTorch bridges
# ─────────────────────────────────────────────────────────────────────────────
function julia_to_torch(field::Array{Float64,3}; device::String="cpu")
    t = torch.from_numpy(np.array(Float32.(field)))
    t = t.unsqueeze(0).unsqueeze(0)
    t = t.to(torch.device(device))
    return t.contiguous()
end

function julia_to_torch_mc(field::Array{Float64,4}; device::String="cpu")
    t = torch.from_numpy(np.array(Float32.(field)))
    t = t.unsqueeze(0)
    t = t.to(torch.device(device))
    return t.contiguous()
end

function torch_to_julia(t)::Array{Float64,3}
    arr = pyconvert(Array{Float32,5}, t.detach().cpu().contiguous().numpy())
    return Float64.(arr[1, 1, :, :, :])
end

function torch_to_julia_mc(t)::Array{Float64,4}
    arr = pyconvert(Array{Float32,5}, t.detach().cpu().contiguous().numpy())
    return Float64.(arr[1, :, :, :, :])
end

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers — POOLED ACROSS BOTH RESOLUTIONS (Run #2 fix #2)
# ─────────────────────────────────────────────────────────────────────────────
"""
    combined_zscore_stats(fields::Array{Float64,3}...)

Pool voxel values from all given fields (e.g. the 128³ AND 192³ DNS fields
for the same physical quantity) and return (mean, std) over the union. Use
this instead of computing stats from the 128³ input alone — Run #1 showed
the true 192³ distribution can be shifted/wider than the 128³ one, so
normalising only from the coarse field risks pushing training targets
outside a well-conditioned numeric range.
"""
function combined_zscore_stats(fields::Array{Float64,3}...)
    pooled = reduce(vcat, (vec(f) for f in fields))
    m, s = mean(pooled), std(pooled)
    s < 1e-12 && (s = 1.0)
    return m, s
end

"""
    combined_scale_stat(fields::Array{Float64,3}...)

Pool the max absolute value across all given fields — use for scale-only
(non-negative) normalisation such as q_l, again pooled across resolutions.
"""
function combined_scale_stat(fields::Array{Float64,3}...)
    return maximum(maximum(f) for f in fields)
end

# ─────────────────────────────────────────────────────────────────────────────
# Julia-side spectral utilities
# ─────────────────────────────────────────────────────────────────────────────
"""Spectral zero-padding 128³ → 192³. RUN #2: reporting/comparison baseline
ONLY — no longer used as a training target (see header note #1)."""
function spectral_zero_pad(field::Array{Float64,3}, N_target::Int)
    N_src = size(field, 1)
    F_src = fft(field)
    half  = N_src ÷ 2
    F_tgt = zeros(ComplexF64, N_target, N_target, N_target)
    for kz in 1:N_src, ky in 1:N_src, kx in 1:N_src
        tx = kx <= half ? kx : N_target - N_src + kx
        ty = ky <= half ? ky : N_target - N_src + ky
        tz = kz <= half ? kz : N_target - N_src + kz
        F_tgt[tx, ty, tz] = F_src[kx, ky, kz]
    end
    scale = (N_target / N_src)^3
    return real.(ifft(F_tgt .* scale))
end

# ─────────────────────────────────────────────────────────────────────────────
# Build FlowFNO model (Python) — IN=OUT=2 channels: [Se, q_l]
# ─────────────────────────────────────────────────────────────────────────────
function build_flowfno_model(;
    in_channels::Int          = 2,
    out_channels::Int         = 2,
    modes::Int                = 6,
    width::Int                = 4,
    n_layers::Int             = 2,
    base_ch::Int              = 2,
    time_dim::Int             = 32,
    noise_scale::Vector{Float64} = fill(0.1, out_channels),
    N_out::Int                = 192,
    device::String            = "cpu"
)
    py_run("""
import torch, torch.nn as nn, torch.nn.functional as F

class SpectralConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.out_ch = out_ch
        self.modes  = modes
        s = 1.0 / (in_ch * out_ch)
        self.weights = nn.ParameterList([
            nn.Parameter(s * torch.randn(in_ch, out_ch, modes, modes, modes, 2))
            for _ in range(4)
        ])
    def cmul(self, x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x,
                            torch.view_as_complex(w.contiguous()))
    def forward(self, x):
        B,C,X,Y,Z = x.shape; m = self.modes
        xf = torch.fft.rfftn(x.contiguous(), dim=[-3,-2,-1])
        of = torch.zeros(B,self.out_ch,X,Y,Z//2+1,dtype=torch.cfloat,device=x.device)
        of[:,:, :m, :m,:m] = self.cmul(xf[:,:, :m, :m,:m],self.weights[0])
        of[:,:,-m:, :m,:m] = self.cmul(xf[:,:,-m:, :m,:m],self.weights[1])
        of[:,:, :m,-m:,:m] = self.cmul(xf[:,:, :m,-m:,:m],self.weights[2])
        of[:,:,-m:,-m:,:m] = self.cmul(xf[:,:,-m:,-m:,:m],self.weights[3])
        return torch.fft.irfftn(of, s=(X,Y,Z), dim=[-3,-2,-1])

class FNO3d_MultiIn(nn.Module):
    def __init__(self, in_ch, out_ch, modes, width, n_layers, N_out):
        super().__init__()
        self.width    = width
        self.n_layers = n_layers
        self.N_out    = N_out
        self.fc0      = nn.Conv3d(in_ch, width, 1)
        self.spec     = nn.ModuleList([SpectralConv3d(width,width,modes) for _ in range(n_layers)])
        self.conv     = nn.ModuleList([nn.Conv3d(width,width,1)          for _ in range(n_layers)])
        self.proj1    = nn.Conv3d(width, width*4, 1)
        self.proj2    = nn.Conv3d(width*4, out_ch, 1)

    def _pad(self, x, N):
        B,C,Nx,Ny,Nz = x.shape
        hx,hy,hz = Nx//2, Ny//2, Nz//2
        xf = torch.fft.rfftn(x.contiguous(), dim=[-3,-2,-1])
        of = torch.zeros(B,C,N,N,N//2+1,dtype=torch.cfloat,device=x.device)
        of[:,:, :hx, :hy,:hz] = xf[:,:, :hx, :hy,:hz]
        of[:,:,-hx:, :hy,:hz] = xf[:,:,-hx:, :hy,:hz]
        of[:,:, :hx,-hy:,:hz] = xf[:,:, :hx,-hy:,:hz]
        of[:,:,-hx:,-hy:,:hz] = xf[:,:,-hx:,-hy:,:hz]
        sc = (N/Nx)**3
        return torch.fft.irfftn(of*sc, s=(N,N,N), dim=[-3,-2,-1])

    def forward(self, x):
        x = F.gelu(self.fc0(x.contiguous()))
        x = self._pad(x, self.N_out)
        for i in range(self.n_layers):
            x = F.gelu(self.spec[i](x) + self.conv[i](x))
        return self.proj2(F.gelu(self.proj1(x)))

class FlowUNet3d(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=2, time_dim=32):
        super().__init__()
        self.base_ch = base_ch
        self.time_mlp  = nn.Sequential(nn.Linear(1,time_dim), nn.GELU(),
                                        nn.Linear(time_dim,time_dim))
        self.time_proj = nn.Linear(time_dim, base_ch*2)
        self.enc1 = self._blk(in_ch,     base_ch)
        self.enc2 = self._blk(base_ch,   base_ch*2)
        self.pool = nn.AvgPool3d(2)
        self.bot  = self._blk(base_ch*2, base_ch*2)
        self.up2  = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = self._blk(base_ch*4, base_ch)
        self.dec1 = self._blk(base_ch*2, base_ch)
        self.out  = nn.Conv3d(base_ch, out_ch, 1)

    def _blk(self, ic, oc):
        ng = max(1, oc//2)
        return nn.Sequential(
            nn.Conv3d(ic,oc,3,padding=1,padding_mode='circular'),
            nn.GroupNorm(ng,oc), nn.GELU(),
            nn.Conv3d(oc,oc,3,padding=1,padding_mode='circular'),
            nn.GroupNorm(ng,oc), nn.GELU())

    def forward(self, x, t_cont):
        tp = self.time_proj(self.time_mlp(t_cont))
        e1 = self.enc1(x.contiguous())
        e2 = self.enc2(self.pool(e1)) + tp[:,:,None,None,None]
        b  = self.bot(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d2u = F.interpolate(d2, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2u, e1], dim=1))
        return self.out(d1)

class FlowSchedule:
    \"\"\"Linear stochastic interpolant / rectified flow, PER-CHANNEL noise
       scale (Run #2 change): noise_scale is now a length-out_ch vector,
       broadcast as (1,C,1,1,1), so Se and q_l can each get the amount of
       stochastic spread their own fluctuation scale needs, tuned
       independently via the std-ratio diagnostic printed after
       reconstruction.\"\"\"
    def __init__(self, noise_scale, device):
        self.noise_scale = torch.tensor(list(noise_scale), dtype=torch.float32,
                                        device=device).view(1,-1,1,1,1)
        self.device = device

    def make_base(self, x_cond):
        z = torch.randn_like(x_cond)
        return x_cond + self.noise_scale * z

    def interpolate(self, x1, x_cond, t):
        x0  = self.make_base(x_cond)
        tb  = t.view(-1,1,1,1,1)
        x_t = (1.0 - tb) * x0 + tb * x1
        v   = x1 - x0
        return x_t, v

    @torch.no_grad()
    def sample(self, net, x_cond, n_steps, heun=True):
        x  = self.make_base(x_cond)
        B  = x_cond.shape[0]
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0  = i * dt
            tt0 = torch.full((B,1), t0, device=self.device, dtype=torch.float32)
            v0  = net(torch.cat([x, x_cond], dim=1), tt0)
            if heun and i < n_steps - 1:
                x_pred = x + dt * v0
                t1  = (i + 1) * dt
                tt1 = torch.full((B,1), t1, device=self.device, dtype=torch.float32)
                v1  = net(torch.cat([x_pred, x_cond], dim=1), tt1)
                x   = x + 0.5 * dt * (v0 + v1)
            else:
                x = x + dt * v0
        return x
""")

    dev  = torch.device(device)
    fno  = py_get("FNO3d_MultiIn")(in_channels, out_channels,
                                    modes, width, n_layers, N_out).to(dev)
    unet = py_get("FlowUNet3d")(2*out_channels, out_channels, base_ch, time_dim).to(dev)
    sched = py_get("FlowSchedule")(pylist(noise_scale), dev)

    n_fno  = pyconvert(Int, sum(p.numel() for p in fno.parameters()))
    n_unet = pyconvert(Int, sum(p.numel() for p in unet.parameters()))
    @info "FNO parameters     : $n_fno"
    @info "Velocity-net params: $n_unet"
    return fno, unet, sched
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-channel weighted MSE loss (Run #2 fix #3)
# ─────────────────────────────────────────────────────────────────────────────
"""
    build_channel_mse_loss(device, out_channels)

Computes MSE PER OUTPUT CHANNEL and combines with per-channel weights,
instead of one pooled MSE over the whole multi-channel tensor. Reused for
BOTH the FNO reconstruction loss and the flow-matching velocity loss (pass
different weight vectors to each call).

Run #1 used one combined `F.mse_loss(pred, tgt)` over both channels. Because
q_l is sparse/near-zero almost everywhere while Se varies substantially at
nearly every voxel, Se's squared error dominated the shared gradient and q_l
collapsed to a near-trivial near-mean prediction. Per-channel loss terms
prevent one channel's error scale from silently out-competing the other's.

Returns `(total_weighted_loss_py, per_channel_mse::Vector{Float64})`.
"""
function build_channel_mse_loss(device::String, out_channels::Int)
    py_run("""
import torch.nn.functional as _F
def _channel_mse_loss(pred, tgt, w):
    per_ch = []
    total = 0.0
    for c in range(pred.shape[1]):
        m = _F.mse_loss(pred[:, c:c+1], tgt[:, c:c+1])
        per_ch.append(m)
        total = total + float(w[c]) * m
    return total, per_ch
""")
    fn_py = py_get("_channel_mse_loss")
    function loss_fn(pred, tgt, weights::Vector{Float64})
        total, per_ch = fn_py(pred, tgt, pylist(weights))
        per_ch_vals = Float64[pyconvert(Float64, m.item()) for m in per_ch]
        return total, per_ch_vals
    end
    return loss_fn
end

# ─────────────────────────────────────────────────────────────────────────────
# FlowFNO struct
# ─────────────────────────────────────────────────────────────────────────────
mutable struct FlowFNO
    params::DomainParams
    device::String
    fno_model::Py
    unet_model::Py
    schedule::Py
    loss_fn::Function
    fno_optimizer::Py
    flow_optimizer::Py
    fno_scheduler::Py
    flow_scheduler::Py
    trained::Bool
    flow_steps::Int
    noise_scale::Vector{Float64}      # per-channel (Run #2)
    in_channels::Int
    out_channels::Int
    fno_channel_weights::Vector{Float64}   # per-channel weights, recon loss (Run #2)
    flow_channel_weights::Vector{Float64}  # per-channel weights, flow loss  (Run #2)
    # Normalisation stats — pooled across 128³ AND 192³ (Run #2 fix #2)
    # channel order throughout: [1]=Se (z-score), [2]=q_l (scale-only)
    Se_mean::Float64;  Se_std::Float64
    ql_scale::Float64
end

function FlowFNO(;
    device::String        = "cpu",
    in_channels::Int      = 2,
    out_channels::Int     = 2,
    fno_modes::Int        = 6,
    fno_width::Int        = 4,
    fno_layers::Int       = 2,
    base_ch::Int          = 2,
    time_dim::Int         = 32,
    noise_scale::Union{Nothing,Vector{Float64}}          = nothing,
    flow_steps::Int       = 20,
    lr_fno::Float64       = 1e-3,
    lr_flow::Float64      = 1e-4,
    epochs::Int           = 300,
    params::DomainParams  = DomainParams(),
    fno_channel_weights::Union{Nothing,Vector{Float64}}  = nothing,
    flow_channel_weights::Union{Nothing,Vector{Float64}} = nothing
)
    ns  = noise_scale          === nothing ? fill(0.1, out_channels) : noise_scale
    fcw = fno_channel_weights  === nothing ? ones(out_channels)      : fno_channel_weights
    flw = flow_channel_weights === nothing ? ones(out_channels)      : flow_channel_weights
    @assert length(ns)  == out_channels "noise_scale must have length out_channels"
    @assert length(fcw) == out_channels "fno_channel_weights must have length out_channels"
    @assert length(flw) == out_channels "flow_channel_weights must have length out_channels"

    init_python!()
    @info "Building FlowFNO (joint Se+q_l, FNO + flow matching, Run #2)..."
    fno, unet, sched = build_flowfno_model(;
        in_channels  = in_channels,
        out_channels = out_channels,
        modes        = fno_modes,
        width        = fno_width,
        n_layers     = fno_layers,
        base_ch      = base_ch,
        time_dim     = time_dim,
        noise_scale  = ns,
        N_out        = params.N_fine,
        device       = device
    )
    @info "Building per-channel MSE loss (Se, q_l weighted separately)..."
    loss_fn = build_channel_mse_loss(device, out_channels)
    fno_opt  = optim.Adam(fno.parameters(),  lr=lr_fno)
    flow_opt = optim.Adam(unet.parameters(), lr=lr_flow)
    fno_sch  = optim.lr_scheduler.CosineAnnealingLR(
                   fno_opt,  T_max=pyint(epochs), eta_min=1e-5)
    flow_sch = optim.lr_scheduler.CosineAnnealingLR(
                   flow_opt, T_max=pyint(epochs), eta_min=1e-6)
    return FlowFNO(params, device, fno, unet, sched, loss_fn,
                   fno_opt, flow_opt, fno_sch, flow_sch,
                   false,
                   flow_steps, ns,
                   in_channels, out_channels,
                   fcw, flw,
                   0.0, 1.0, 1.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# Joint training
# ─────────────────────────────────────────────────────────────────────────────
"""
    train_flowfno!(pipeline, inputs_list, targets_list;
                   epochs, fno_weight, flow_weight, log_every)

`targets_list` should now be built from the TRUE 192³ DNS fields (Run #2 —
see `augment_dns` in SR_Cloud_Example.jl), not a spectral-padding baseline.

Per-channel MSE (Se, q_l) is logged separately for BOTH the FNO and flow
branches every `log_every` epochs, so channel imbalance (like the q_l
collapse found in Run #1) is visible during training, not just at the end.
"""
function train_flowfno!(pipeline::FlowFNO,
                        inputs_list::Vector{<:AbstractArray},
                        targets_list::Vector{<:AbstractArray};
                        epochs::Int          = 300,
                        fno_weight::Float64  = 1.0,
                        flow_weight::Float64 = 1.0,
                        log_every::Int       = 30)
    dev     = pipeline.device
    fno     = pipeline.fno_model
    unet    = pipeline.unet_model
    sched   = pipeline.schedule
    loss_fn = pipeline.loss_fn
    n       = length(inputs_list)
    C       = pipeline.out_channels

    @info "Training FlowFNO on $n samples for $epochs epochs " *
          "(noise_scale=$(pipeline.noise_scale), " *
          "fno_weights=$(pipeline.fno_channel_weights), " *
          "flow_weights=$(pipeline.flow_channel_weights))"

    for epoch in 1:epochs
        ep_fno_ch  = zeros(Float64, C)
        ep_flow_ch = zeros(Float64, C)

        for i in randperm(n)
            x_mc = julia_to_torch_mc(inputs_list[i]; device=dev)
            tgt  = julia_to_torch_mc(targets_list[i]; device=dev)

            field_mean = fno(x_mc)
            fno_total, fno_mse_ch = loss_fn(field_mean, tgt, pipeline.fno_channel_weights)

            t = torch.rand(pytuple((1, 1))).to(torch.device(dev))
            field_mean_det = field_mean.detach()

            xt_v  = sched.interpolate(tgt, field_mean_det, t)
            x_t   = xt_v[0]
            v_tgt = xt_v[1]

            x_unet = torch.cat(pylist([x_t, field_mean_det]), dim=1)
            v_pred = unet(x_unet, t)
            flow_total, flow_mse_ch = loss_fn(v_pred, v_tgt, pipeline.flow_channel_weights)

            total = fno_weight * fno_total + flow_weight * flow_total
            pipeline.fno_optimizer.zero_grad()
            pipeline.flow_optimizer.zero_grad()
            total.backward()
            pipeline.fno_optimizer.step()
            pipeline.flow_optimizer.step()

            ep_fno_ch  .+= fno_mse_ch
            ep_flow_ch .+= flow_mse_ch
        end

        pipeline.fno_scheduler.step()
        pipeline.flow_scheduler.step()

        if epoch % log_every == 0
            ep_fno_ch  ./= n
            ep_flow_ch ./= n
            @printf("  FlowFNO Epoch %4d/%d | FNO MSE  Se=%.6f ql=%.6f | Flow MSE  Se=%.6f ql=%.6f\n",
                    epoch, epochs, ep_fno_ch[1], ep_fno_ch[2], ep_flow_ch[1], ep_flow_ch[2])
        end
    end

    pipeline.trained = true
    @info "FlowFNO training complete."
end

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction
# ─────────────────────────────────────────────────────────────────────────────
"""
    reconstruct(pipeline, Se_128, ql_128;
                n_flow_steps, heun, clamp_ql_negative)

Returns `(Se_mean, ql_mean, Se_sample, ql_sample)`, all Array{Float64,3} at
192³, in physical units. Only q_l is clamped to ≥ 0 — Se is signed.
"""
function reconstruct(pipeline::FlowFNO,
                     Se_128::Array{Float64,3},
                     ql_128::Array{Float64,3};
                     n_flow_steps::Int        = pipeline.flow_steps,
                     heun::Bool               = true,
                     clamp_ql_negative::Bool  = true)
    @assert pipeline.trained "Run train_flowfno! first."

    dev = pipeline.device

    Se_n = (Se_128 .- pipeline.Se_mean) ./ pipeline.Se_std
    ql_n = ql_128 ./ pipeline.ql_scale

    x_in = zeros(Float64, 2, size(Se_n)...)
    x_in[1,:,:,:] = Se_n
    x_in[2,:,:,:] = ql_n

    x_mc = julia_to_torch_mc(x_in; device=dev)

    pipeline.fno_model.eval()
    pipeline.unet_model.eval()
    local field_mean_t, field_sample_t

    let ctx = torch.no_grad()
        ctx.__enter__()
        try
            @info "FlowFNO: FNO inference → joint mean [Se, q_l]..."
            field_mean_t = pipeline.fno_model(x_mc)
            @info "FlowFNO: ODE integration ($n_flow_steps steps, heun=$heun)..."
            field_sample_t = pipeline.schedule.sample(
                pipeline.unet_model, field_mean_t, pyint(n_flow_steps), heun)
        finally
            ctx.__exit__(nothing, nothing, nothing)
        end
    end

    field_mean   = torch_to_julia_mc(field_mean_t)
    field_sample = torch_to_julia_mc(field_sample_t)

    Se_mean   = field_mean[1,:,:,:]   .* pipeline.Se_std .+ pipeline.Se_mean
    ql_mean   = field_mean[2,:,:,:]   .* pipeline.ql_scale
    Se_sample = field_sample[1,:,:,:] .* pipeline.Se_std .+ pipeline.Se_mean
    ql_sample = field_sample[2,:,:,:] .* pipeline.ql_scale

    if clamp_ql_negative
        ql_mean   .= max.(ql_mean,   0.0)
        ql_sample .= max.(ql_sample, 0.0)
    end

    return Se_mean, ql_mean, Se_sample, ql_sample
end

# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────
function save_models(pipeline::FlowFNO, dir::String)
    mkpath(dir)
    torch.save(pipeline.fno_model.state_dict(),
               joinpath(dir, "flowfno_fno.pt"))
    torch.save(pipeline.unet_model.state_dict(),
               joinpath(dir, "flowfno_velocity.pt"))
    @info "FlowFNO weights saved to $dir/"
end

function load_models!(pipeline::FlowFNO, dir::String)
    dev = torch.device(pipeline.device)
    fp  = joinpath(dir, "flowfno_fno.pt")
    up  = joinpath(dir, "flowfno_velocity.pt")
    if isfile(fp)
        pipeline.fno_model.load_state_dict(torch.load(fp, map_location=dev))
        @info "FNO weights loaded."
    else @warn "FNO weights not found at $fp" end
    if isfile(up)
        pipeline.unet_model.load_state_dict(torch.load(up, map_location=dev))
        pipeline.trained = true
        @info "Velocity-net weights loaded."
    else @warn "Velocity-net weights not found at $up" end
end

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
println("""
╔══════════════════════════════════════════════════════════════╗
║  FlowFNO: FNO + Flow Matching / Stochastic Interpolant       ║
║          Joint Se + q_l  (128³ → 192³)  —  RUN #2            ║
║  Joint FNO + velocity-net — single backward pass per step    ║
║                                                              ║
║  Inputs : [Se, q_l] at 128³  (DNS Case_H_P_128)              ║
║  Target : [Se, q_l] at TRUE 192³ DNS  (DNS Case_H_P_192)     ║
║           (Run #1 mistakenly trained against a spectral      ║
║            zero-padding of the 128³ field — fixed here)      ║
║  Output : field_mean (FNO) + field_sample (flow matching)    ║
║                                                              ║
║  Sampler: few-step ODE (Heun) from the FNO mean, not noise   ║
║  Loss   : per-channel weighted MSE (Se, q_l weighted         ║
║           separately — Run #1 let Se dominate a shared MSE)  ║
║  Norm.  : Se/q_l stats pooled across BOTH resolutions        ║
║  Memory : ~7.5-8.6 GB total — near the 8 GB ceiling          ║
╚══════════════════════════════════════════════════════════════╝
""")
