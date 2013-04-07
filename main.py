#!/usr/bin/env python2
import sys
import os

def main():
    if len(sys.argv) == 1:
        for root, dirs, files in os.walk('.'):
            if root == '.':
                break
        try:
            dirs.remove('.git')
        except ValueError:
            pass
    else:
        dirs = sys.argv[1:]
    for package in sorted(dirs):
        pack = __import__(package + ".test")
        pack.test.test()


if __name__ == '__main__':
    main()
