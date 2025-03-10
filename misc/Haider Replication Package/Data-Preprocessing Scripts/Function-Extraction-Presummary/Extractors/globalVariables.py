debug = 1
LEN = 512 * 4
class Info:
    def __init__(self):
        self.output = ''
        self.start = (0, 0)
        self.end = (0, 0)

def intersect(start1, end1, start2, end2):
    if start1[0] > end2[0] or start2[0] > end1[0]:
        return False
    # if start1[1] > end2[1] or start2[1] > end1[1]:
    #     return False
    return True
