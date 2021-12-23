def random_only(epoch):
    return 'random', 'none'

def random_align(epoch):
    return 'random', 'align'

def align_only(epoch):
    return 'align', 'none'

def init_align_then_random(epoch):
    if epoch <= 5:
        return 'align', 'none'
    else:
        return 'random', 'align'