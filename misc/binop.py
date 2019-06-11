# Build binary_ops.cc with -DTIMER
# VE_LOG_LEVEL=1 python ... 2> log
# python binop.py log

import sys
import re

def parse(f):
    ary = []
    while True:
        line = f.readline()
        if not line:
            break
        m = re.match(r'op_Binary::(\w+): (in.*)', line)
        if m:
            op = m.group(1)
            param = m.group(2)
            param = re.sub('dtype=1,', '', re.sub('nelems=', 'n=', param))
            while True:
                line = f.readline()
                #print(line)
                m = re.match(r'op_Binary::{}: ([0-9\.]+) msec'.format(op), line)
                if m:
                    time = float(m.group(1))
                    tmp = {'op': op, 'time': time, 'param': param}
                    ary.append(tmp)
                    break
    return ary

def main():
    filename = sys.argv[1]
    with open(filename) as f:
        ary = parse(f)
        data = {}
        for tmp in ary:
            #print('{:8.3f} {:20} {}'.format(tmp['time'], tmp['op'], tmp['param']))
            op = tmp['op']
            param = tmp['param']
            if not op in data:
                data[op] = {}
            if not param in data[op]:
                data[op][param] = {}
                data[op][param]['cnt'] = 0
                data[op][param]['time'] = 0
            data[op][param]['time'] += tmp['time']
            data[op][param]['cnt'] += 1
        for op in data:
            for param in data[op]:
                d = data[op][param]
                ave = d['time'] / d['cnt']
                print('{:8.3f} {:8.3f} {:8} {:20} {}'.format(d['time'], ave, d['cnt'], op, param))


if __name__ == "__main__":
    main()
