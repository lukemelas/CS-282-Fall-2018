import pdb

class AverageMeter(object):
    """Computes and stores the average and current value. 
       Also stores a rolling average of size roll_len."""
    def __init__(self, roll_len=100):
        self.reset()
        self.roll_len = roll_len
        self.roll = []

    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0; self.roll = []; self.roll_avg = 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count
        self.roll = (self.roll + [val] * n)[-self.roll_len:]
        self.roll_avg = sum(self.roll) / len(self.roll)
        
def check(val, msg):
    '''Checks whether val is nan or inf and prints msg if True'''
    if not val: print(msg); pdb.set_trace()
        
def to_numpy(t):
    '''PyTorch tensor to numpy array'''
    return t.detach().to('cpu').numpy()
