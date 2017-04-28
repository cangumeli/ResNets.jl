import matplotlib.pyplot as plt

files = [
    ('ResNet110', 'train_resnet110.out'),
    ('ResNet32', 'train_resnet32.out')
]
for tf in files:
    title, fname = tf
    with open(fname) as f:
        lines = filter(lambda x: x.startswith('(:iter'), f.readlines())
        iters = []
        trns = []
        tsts = []
        for l in lines:
            vals = l.split(',')
            iters.append(int(vals[1]))
            trns.append(1 - float(vals[3]
                .replace('(', '')
                .replace('f0', '')))
            tsts.append(1 - float(vals[6]
                .replace('(', '')
                .replace('f0', '')))
        trns = [a*100 for a in trns]
        tsts = [a*100 for a in tsts]
        print iters
        print " \n....\n"
        print trns
        print " \n....\n"
        print tsts
        # Compute the peak accuracy
        min_trn, min_tst = 101, 101
        for (i, t) in enumerate(trns):
            if t < min_trn:
                min_trn = t
                min_tst = tsts[i]
        print "Peak training error rate %.2f"%(min_trn)
        print "Peak test error rate %.2f"%(min_tst)
        plt.plot(iters, tsts, label='test error')
        plt.plot(iters, trns, label='training error')
        plt.legend()
        plt.title(title)
        plt.xlim(0, iters[-1])
        plt.ylim(0, 100)
        plt.yticks([i*10 for i in xrange(1, 10)])
        # plt.axis([0, iters[-1], 0, 100])
        plt.xlabel('iters')
        plt.ylabel('% error')
        plt.show()
