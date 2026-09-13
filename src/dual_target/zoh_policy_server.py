"""SmolVLA-only worker. Receives no geometry, GT, scorer flags or episode IDs."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

from .contracts import IMAGE_KEY,STATE_KEY,validate_policy_input
from .zoh_policy_contract import validate_action_chunk
from .tiny_train_core import file_sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    output=sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from PIL import Image
        from .zoh_train import load_training_policy,reload_processors
        manifest=json.loads((args.checkpoint/'checkpoint_manifest.json').read_text())
        for name,digest in manifest['files_sha256'].items():
            if file_sha(args.checkpoint/name)!=digest:
                raise ValueError('checkpoint hash mismatch '+name)
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        model=load_training_policy(args.checkpoint/'pretrained_model').eval()
        pre,post=reload_processors(args.checkpoint/'pretrained_model')
        config=model.config
        if (config.chunk_size!=5 or config.n_action_steps!=1 or config.input_features[STATE_KEY].shape!=(3,)
                or config.action_feature.shape!=(3,)):
            raise ValueError('wrong model interface')
        generator=torch.Generator(device='cuda').manual_seed(args.seed)
    output.write(json.dumps({'status':'ready','checkpoint_manifest_sha256':file_sha(args.checkpoint/'checkpoint_manifest.json'),
        'precision':'CUDA BF16 autocast, FP32 trainable parameters'})+'\n')
    output.flush()
    for line in sys.stdin:
        request=validate_policy_input(json.loads(line))
        with contextlib.redirect_stdout(sys.stderr):
            image=np.asarray(Image.open(request[IMAGE_KEY]).convert('RGB')).copy()
            if image.shape!=(512,512,3):
                raise ValueError('wrong input RGB shape')
            batch=pre({IMAGE_KEY:torch.from_numpy(image).permute(2,0,1).float()/255,
                STATE_KEY:torch.tensor(request[STATE_KEY],dtype=torch.float32),'task':request['task']})
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                noise=torch.randn((1,5,32),device='cuda',generator=generator)
                action=post(model.predict_action_chunk(batch,noise=noise)).float()[0].tolist()
            validate_action_chunk(action)
        output.write(json.dumps({'actions':action},allow_nan=False)+'\n')
        output.flush()


if __name__=='__main__':
    main()
