"""Pre-specified horizon metrics with censoring distribution fitted on train."""
import numpy as np


def harrell_c(time,event,risk):
    time,event,risk=(np.asarray(x) for x in (time,event,risk))
    if not (time.ndim == event.ndim == risk.ndim == 1 and len(time) == len(event) == len(risk)):
        raise ValueError('Expected equal-length vectors')
    if not all(np.isfinite(x).all() for x in (time,event,risk)) or not np.isin(event,[0,1]).all():
        raise ValueError('Invalid metric input')
    good=0.0
    total=0
    for i in range(len(time)):
        valid=(time > time[i]) & (event[i] == 1)
        total+=int(valid.sum())
        good+=float((risk[i] > risk[valid]).sum()) + .5*float((risk[i] == risk[valid]).sum())
    return {'value':good/total if total else None,'comparable_pairs':total}


def censor_km(train_time,train_event):
    """Kaplan-Meier G for censoring; event observed is non-censoring."""
    t,e=(np.asarray(x) for x in (train_time,train_event))
    if t.ndim != 1 or e.shape != t.shape or len(t)==0 or not np.isfinite(t).all() or (t<0).any() or not np.isin(e,[0,1]).all():
        raise ValueError('Invalid training survival data')
    unique=np.unique(t)
    g=1.0
    steps=[]
    for point in unique:
        at_risk=int((t>=point).sum())
        censored=int(((t==point)&(e==0)).sum())
        g*=1-censored/at_risk
        steps.append((float(point),float(g)))
    return steps


def _g_at(steps,t,left=False):
    result=1.0
    for point,g in steps:
        if point > t or (left and point == t):
            break
        result=g
    return result


def ipcw_brier(train_time,train_event,test_time,test_event,survival,horizon,eps=1e-6):
    t,e,s=(np.asarray(x) for x in (test_time,test_event,survival))
    if t.ndim != 1 or e.shape != t.shape or s.shape != t.shape or len(t)==0 or horizon<0:
        raise ValueError('Invalid test metric inputs')
    if not all(np.isfinite(x).all() for x in (t,e,s)) or (t<0).any() or not np.isin(e,[0,1]).all() or ((s<0)|(s>1)).any():
        raise ValueError('Invalid test values')
    steps=censor_km(train_time,train_event)
    g_horizon=_g_at(steps,horizon)
    if g_horizon <= eps:
        return {'value':None,'reason':'Censoring survival at horizon too small','g_horizon':g_horizon}
    total=0.0
    for ti,ei,si in zip(t,e,s):
        if ti <= horizon and ei == 1:
            g=_g_at(steps,ti,left=True)
            if g<=eps:
                return {'value':None,'reason':'Censoring survival before event too small','g_horizon':g_horizon}
            total+=si**2/g
        elif ti > horizon:
            total+=(1-si)**2/g_horizon
    return {'value':float(total/len(t)),'g_horizon':g_horizon,
            'at_risk':int((t>horizon).sum()),'n':len(t)}
