import json
from pathlib import Path
import pandas as pd
from commercereclab.evaluation.stage import _stage_queries, evaluate_stage_aware, write_report


def _dataset(root: Path):
    d=root/'retailrocket'; d.mkdir()
    rows=[]
    # early source-fit sessions
    for v,t,item in [(1,1000,11),(2,1200,12),(3,1400,11),(4,1600,12)]:
        rows += [[t,v,'view',10,None],[t+10,v,'view',item,None],[t+20,v,'addtocart',item,None],[t+40,v,'transaction',item,v]]
    # later train, validation, test
    for v,t,item in [(5,3000,11),(6,3300,12),(7,5000,11),(8,7000,12)]:
        rows += [[t,v,'view',10,None],[t+10,v,'view',item,None],[t+20,v,'addtocart',item,None],[t+40,v,'transaction',item,v]]
    pd.DataFrame(rows,columns=['timestamp','visitorid','event','itemid','transactionid']).to_csv(d/'events.csv',index=False)
    props=pd.DataFrame([[900,10,'categoryid','1'],[900,11,'categoryid','1'],[900,12,'categoryid','1']],columns=['timestamp','itemid','property','value'])
    props.iloc[:2].to_csv(d/'item_properties_part1.csv',index=False); props.iloc[2:].to_csv(d/'item_properties_part2.csv',index=False)
    pd.DataFrame({'categoryid':[1],'parentid':[None]}).to_csv(d/'category_tree.csv',index=False)
    return d


def _manifest(root):
    p=root/'manifest.json'; p.write_text(json.dumps({'train_end_ms':4000,'validation_end_ms':6000,'session_gap_minutes':30,'session_boundary_policy':'split_boundary_breaks_session','prediction_horizon':'remainder_of_current_split_bounded_session'})); return p


def test_stage_queries_create_cart_stage():
    df=pd.DataFrame([[1,1,'view',10,'validation'],[2,1,'addtocart',11,'validation'],[3,1,'transaction',11,'validation']],columns=['timestamp','visitorid','event','itemid','split'])
    q,_=_stage_queries(df,split='validation',gap_minutes=30,max_sessions=None,seed=1)
    assert [x['stage'] for x in q] == ['view_stage','cart_stage']
    assert q[1]['positives']['transaction'].tolist()==[11]


def test_stage_aware_report(tmp_path: Path):
    d=_dataset(tmp_path); m=_manifest(tmp_path)
    r=evaluate_stage_aware(d,m,k=1,category_depth=3,behavioral_depth=1,internal_fit_fraction=.5,max_internal_ranker_sessions=None,max_sessions_per_split=None,max_nonrelevant_per_query=3)
    assert 'cart_stage' in r.training_examples
    assert 'stage_conditioned' in r.metrics['validation']
    assert r.prediction_points['view_stage'] == 'after_first_event_only_when_first_event_is_view'
    jp,mp=write_report(r,tmp_path/'out')
    assert 'validation_retain_stage_conditioning' in json.loads(jp.read_text())
    text=mp.read_text()
    assert 'Stage-Conditional Ranking' in text
    assert 'only when that observed first event is a view' in text
