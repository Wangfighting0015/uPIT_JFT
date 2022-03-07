from pystoi import stoi
from pypesq import pesq
import torch
from mir_eval.separation import bss_eval_sources

def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    sdr, _, _, _ = bss_eval_sources(egs, est)
    mix_sdr, _, _, _ = bss_eval_sources(egs, mix)
    return float(sdr-mix_sdr)

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    EPS =1e-8
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr

#a_tgt ="./cache_baseline/22ga010d_1.5482_052o020t_-1.5482.wav.spk1.wav"
#a_mix ="/media/dell/wtt_resources/singapore/projects/con_tasNet/Dataset/wsj0-mix/2speakers/wav8k/min/tt/s1/22ga010d_1.5482_052o020t_-1.5482.wav"


av_sisnr =cal_SISNR(a_tgt, a_mix)
#av_sdr =SDR(estimate_source, a_tgt, a_mix)
#av_pesq =pesq(a_tgt, a_mix, 16000)
#av_stoi =stoi(a_tgt, a_mix, 16000, extended=False)