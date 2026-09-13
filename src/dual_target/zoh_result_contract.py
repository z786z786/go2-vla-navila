"""Normalize legacy timeout metadata only when a complete trace proves its length."""


def normalized_result(result, pre_count, post_count, max_steps=1500):
    if pre_count<=0 or pre_count!=post_count:
        raise ValueError('incomplete pre/post evidence')
    out=dict(result)
    if 'steps' not in out:
        if out.get('status')!='FAILED_TIMEOUT' or pre_count!=max_steps:
            raise ValueError('missing steps without a complete max-length timeout trace')
        out['steps']=pre_count
        out['steps_provenance']='legacy timeout: derived from complete equal-length pre/post trace'
    if type(out['steps']) is not int or out['steps']!=pre_count:
        raise ValueError('declared steps disagree with evidence')
    return out
