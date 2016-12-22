import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ks = [10, 15, 20, 25]
scores = []

for k in ks:
    summ = 0.0
    count = 0
    with open('k{}'.format(k), 'r') as f:
        contents = f.readlines()
    for l in contents:
        if l.strip():
            score, precision, recall = [float(s.strip()) for s in l.split(' ')]
            summ += score
            count += 1
    scores.append(summ / count)

print scores

plt.clf()
plt.figure()
plt.plot(ks, scores, "--o", label="avg pos recall")
plt.ylim([0,1])
plt.title("SRW Avg. Recall")
plt.legend(loc='lower right')
plt.xlabel("k values (for core node)")
plt.ylabel("Recall")
plt.savefig("srwAvgResults.png")
