"""Generic spelling, component and numeric evidence for follow-up experiments."""
import re
import numpy as np
from rapidfuzz import fuzz
from .features import jaccard
from .blocking import NAME_STOP

EXTRA_NAMES = ['core_sort','core_set','core_partial','name_partial','address_partial',
               'numeric_canonical_jaccard','numeric_left_coverage','numeric_right_coverage',
               'numeric_first_equal','numeric_last_equal','numeric_count_difference',
               'address_words_sort','address_words_set','skeleton_sort','skeleton_set']


def extra_features(rows):
    result=[]
    for r in rows:
        a,b=r[2],r[6]; c,d=r[3],r[7]
        ca=' '.join(t for t in a.split() if t not in NAME_STOP)
        cb=' '.join(t for t in b.split() if t not in NAME_STOP)
        na=[str(int(n)) for n in re.findall('[0-9]+',c)]
        nb=[str(int(n)) for n in re.findall('[0-9]+',d)]
        sa,sb=set(na),set(nb); both=bool(na and nb)
        wa=re.sub('[0-9]+',' ',c); wb=re.sub('[0-9]+',' ',d)
        ska=' '.join(t[:1]+re.sub('[aeiou]','',t[1:]) for t in ca.split())
        skb=' '.join(t[:1]+re.sub('[aeiou]','',t[1:]) for t in cb.split())
        def sim(fn,x,y): return fn(x,y)/100 if x.strip() and y.strip() else 0.
        result.append([sim(fuzz.token_sort_ratio,ca,cb),sim(fuzz.token_set_ratio,ca,cb),
            sim(fuzz.partial_ratio,ca,cb),sim(fuzz.partial_ratio,a,b),sim(fuzz.partial_ratio,c,d),
            jaccard(sa,sb),len(sa&sb)/max(len(sa),1),len(sa&sb)/max(len(sb),1),
            float(both and na[0]==nb[0]),float(both and na[-1]==nb[-1]),abs(len(na)-len(nb)),
            sim(fuzz.token_sort_ratio,wa,wb),sim(fuzz.token_set_ratio,wa,wb),
            sim(fuzz.token_sort_ratio,ska,skb),sim(fuzz.token_set_ratio,ska,skb)])
    return np.asarray(result,dtype=np.float32).reshape(-1,len(EXTRA_NAMES))
