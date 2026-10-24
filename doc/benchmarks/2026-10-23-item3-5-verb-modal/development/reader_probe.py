import SentenceCredit
original = SentenceCredit.reader_weights
def weights(parts, active, draw):
    result = original(parts, active, draw)
    if draw is not None:
        print('DRAW', {k:v.tolist() if hasattr(v,'tolist') else v for k,v in draw.items()}, 'R+A', (parts[:,:,0]+parts[:,:,2]).tolist())
    return result
SentenceCredit.reader_weights = weights
