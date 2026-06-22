def get_relation_map(dataname):
    base_map = {
        'pa': ('paper', 'pa', 'author'),
        'ap': ('author', 'ap', 'paper'),
    }
    if 'acm' in dataname:
        base_map.update({
            'pf': ('paper', 'pf', 'field'),
            'fp': ('field', 'fp', 'paper'),
        })
    elif 'dblp' in dataname:
        base_map.update({
            'pc': ('paper', 'pc', 'conference'),
            'cp': ('conference', 'cp', 'paper'),
            'pt': ('paper', 'pt', 'term'),
            'tp': ('term', 'tp', 'paper'),
        })
    elif 'aminer' in dataname:
        base_map.update({
            'pr': ('paper', 'pr', 'research'),
            'rp': ('research', 'rp', 'paper'),
        })
    elif 'imdb' in dataname:
        base_map.update({
            'md': ('movie', 'md', 'director'),
            'dm': ('director', 'dm', 'movie'),
            'ma': ('movie', 'ma', 'actor'),
            'am': ('actor', 'am', 'movie'),
        })
    return base_map
