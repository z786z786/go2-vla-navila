import json
from pathlib import Path
from go2_nav.contracts import Action, ActionFilter, validate_episode
from go2_nav.learning import flow_matching_loss, flow_sample, masked_decode
from go2_nav.runtime import ColorBoxPolicy, TwoBoxMock


def test_action_filter_clip_and_slew():
    f = ActionFilter(); out = f.apply(Action(2.0, 2.0), .2)
    assert out.vx <= .35 and abs(out.wz) <= .5


def test_two_box_visual_loop(tmp_path):
    env = TwoBoxMock(tmp_path); policy = ColorBoxPolicy(); obs=env.reset('go to the red box')
    rows=[]
    for _ in range(100):
        action=policy.predict(obs); rows.append({'observation':obs.to_dict(),'action':action.to_dict()})
        obs=env.step(action)
        if action.stop: break
    assert rows[-1]['action']['stop']
    record={'schema_version':1,'episode_id':'t','scene_id':'two_box_mock','source':'mock','policy':'mock','outcome':'stopped','steps':rows}
    validate_episode(record)


def test_learning_primitives():
    v=lambda x,t: [1.0 for _ in x]
    assert flow_sample(v,[0.,0.],2)==[1.,1.]
    assert flow_matching_loss(v,[0.],[1.],1.)==0.
    out=masked_decode(lambda tokens:[(i+1,1.) for i in range(len(tokens))],3,3)
    assert out == [1,2,3]
