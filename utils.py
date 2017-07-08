import sys
import time
import progressive.bar


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
            print(colorize('%s \t' % self.name, 'green'), file=self.output)
        self.tstart = time.time()
        self.output.flush()

    def __exit__(self, type, value, traceback):
        print(colorize('Elapsed: %s' % (time.time() - self.tstart), 'green'), file=self.output)
        self.output.flush()


class ProgressBar(object):
    def __init__(self, max_value, title='', unit='GB', **kwargs):
        self.bar = progressive.bar.Bar(max_value=max_value, title=title, **kwargs)
        self.bar.cursor.clear_lines(2)
        self.bar.cursor.save()
        self.unit = unit

    def update(self, val, max_value=None):
        self.bar.cursor.restore()
        if max_value: self.bar.max_value = max_value
        self.bar.draw(val, newline=False)
        sys.stdout.write(' {}\n'.format(self.unit))


class SpinCursor:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)

