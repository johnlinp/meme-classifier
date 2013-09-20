# -*- coding: utf-8 -*-

import sys
from memeclass import MemeClassifier

if len(sys.argv) == 1:
    print 'usage:'
    print '    python %s image_filename' % sys.argv[0]
    exit()

mc = MemeClassifier('templates')
sys.argv.pop(0)
for img_fname in sys.argv:
    meme_name = mc.classify(img_fname)
    print img_fname, meme_name
