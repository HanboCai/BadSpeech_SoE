import random
import yaml
from munch import Munch
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
import paddleaudio
import librosa
import soundfile as sf

from starganv2vc.starganv2vc_paddle.Utils.ASR.models import ASRCNN
from starganv2vc.starganv2vc_paddle.Utils.JDC.model import JDCNet
from starganv2vc.starganv2vc_paddle.models import Generator, MappingNetwork, StyleEncoder

from yacs.config import CfgNode
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator


def preprocess(wave):
    wave_tensor = paddle.to_tensor(wave).astype(paddle.float32)
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (paddle.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = paddle.to_tensor([speaker], dtype=paddle.int64)
            latent_dim = starganv2.mapping_network.shared[0].weight.shape[0]
            ref = starganv2.mapping_network(paddle.randn([1, latent_dim]), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave)

            with paddle.no_grad():
                label = paddle.to_tensor([speaker], dtype=paddle.int64)
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings
    
    


speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]

to_mel = paddleaudio.features.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
to_mel.fbank_matrix[:] = paddle.load('starganv2vc/starganv2vc_paddle/fbank_matrix.pd')['fbank_matrix']
mean, std = -4, 4
F0_model = JDCNet(num_class=1, seq_len=192)
params = paddle.load("starganv2vc/Models/bst.pd")['net']
F0_model.set_state_dict(params)
_ = F0_model.eval()

with open('starganv2vc/Vocoder/config.yml') as f:
    voc_config = CfgNode(yaml.safe_load(f))
voc_config["generator_params"].pop("upsample_net")
voc_config["generator_params"]["upsample_scales"] = voc_config["generator_params"].pop("upsample_params")["upsample_scales"]
vocoder = PWGGenerator(**voc_config["generator_params"])
vocoder.remove_weight_norm()
vocoder.eval()
vocoder.set_state_dict(paddle.load('starganv2vc/Vocoder/checkpoint-400000steps.pd'))

model_path = 'starganv2vc/Models/vc_ema.pd'

with open('starganv2vc/Models/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = paddle.load(model_path)
params = params['model_ema']
_ = [starganv2[key].set_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.style_encoder = starganv2.style_encoder
starganv2.mapping_network = starganv2.mapping_network
starganv2.generator = starganv2.generator



def loadvcmodel(id_num):
    # 计算Demo文件夹下的说话人的风格
    speaker_dicts = {}
    #selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
    selected_speakers = [id_num]
    for s in selected_speakers:
        k = s
        speaker_dicts['p' + str(s)] = ('starganv2vc/Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))

    reference_embeddings = compute_style(speaker_dicts)
    return reference_embeddings,speaker_dicts
    
def voice_convert(reference_embeddings,speaker_dicts,path,save_path):
    wav_path = path

    audio, source_sr = librosa.load(wav_path, sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32

    import time
    start = time.time()
    source = preprocess(audio)
    keys = []
    converted_samples = {}
    reconstructed_samples = {}
    converted_mels = {}

    for key, (ref, _) in reference_embeddings.items():
        with paddle.no_grad():
            f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
            out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
            
            c = out.transpose([0,1,3,2]).squeeze()
            y_out = vocoder.inference(c)
            y_out = y_out.reshape([-1])

            if key not in speaker_dicts or speaker_dicts[key][0] == "":
                recon = None
            else:
                wave, sr = librosa.load(speaker_dicts[key][0], sr=24000)
                mel = preprocess(wave)
                c = mel.transpose([0,2,1]).squeeze()
                recon = vocoder.inference(c)
                recon = recon.reshape([-1]).numpy()

        converted_samples[key] = y_out.numpy()
        reconstructed_samples[key] = recon

        converted_mels[key] = out
        
        keys.append(key)
    end = time.time()
    print('Total cost time: %.3f sec' % (end - start) )

    for key, wave in converted_samples.items():
        print('Voice convert result: %s' % key)
        sf.write(save_path,wave,24000)