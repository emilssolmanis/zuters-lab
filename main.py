#!/usr/bin/env python2
import sys
import os

def main():
    for root, dirs, files in os.walk('.'):
        if root == '.':
            break
    for package in dirs:
        pack = __import__(package + ".test")
        pack.test.test()


if __name__ == '__main__':
    main()
