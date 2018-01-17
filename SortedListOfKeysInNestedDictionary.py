# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:26:27 2018

@author: msharma
"""

def get_iterator(i):
    if hasattr(i, 'next'):
        # already an iterator - use it as-is!
        return i
    elif hasattr(i, '__iter__') and not isinstance(i, basestring):
        # an iterable type that isn't a string
        return iter(i)
    else: 
        # Can't iterate most other types!
        return None

def get_dict_keys(D):
    LRtn = []
    L = [(D, get_iterator(D))]
    while 1:
        if not L: break
        cur, _iter = L[-1]

        if _iter:
            # Get the next item
            try: 
                i = _iter.next()
            except StopIteration:
                del L[-1]
                continue

        if isinstance(cur, dict): 
            # Process a dict and all subitems
            LRtn.append(i)

            _iter = get_iterator(cur[i])
            if _iter: L.append((cur[i], _iter))
        else:
            # Process generators, lists, tuples and all subitems
            _iter = get_iterator(i)
            if _iter: L.append((i, _iter))

    # Sort and return
    LRtn.sort()
    return LRtn

#D = {
#    'a'         : 'aaaa',
#    'level2_1'  : {'b': 'bbbbb', 'level3': {'c': 'cccc', 'd': 'dddddd', 'e': 134, 'f': [{'blah': 553}]}  },
#    'level2_2'  : { 'z': 'zzzzzzz' },
#    'blah2': iter([{'blah3': None}]),
#}
#
#print get_dict_keys(D)