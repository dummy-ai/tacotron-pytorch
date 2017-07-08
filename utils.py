import sys
import time


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight = False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Timed(object):
    def __init__(self, name=None, output=sys.stdout):
        self.name = name
        if output and type(output) == str:
            self.output = open(output, 'w')
        else:
            self.output = output

    def __enter__(self):
        if self.name:
            print(colorize('[%s] Began \t' % self.name, 'green'), file=self.output)
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(colorize('[%s]\t' % self.name, 'green'), file=self.output)
        print(colorize('Elapsed: %s' % (time.time() - self.tstart), 'green'), file=self.output)
        self.output.flush()

