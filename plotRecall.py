import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

k = [25, 30, 35, 40, 45, 50]

avgPosPrecs = [0.9975807766716708, 0.9955060809935166, 0.9977252230158127, 0.9973235019569081, 0.996184745725351, 0.9944364394234271]
avgPosRecs = [0.842138939054303, 0.8229604047409369, 0.806703236611724, 0.7824716271426965, 0.756226061293934, 0.7247646025696188]

plt.clf()
plt.figure()
plt.plot(k, avgPosPrecs, "--o", label="avg pos precision")
plt.plot(k, avgPosRecs, "--o", label="avg pos recall")
plt.ylim([0,1])
plt.title("Avg. Positive Precision + Recall - No Proximity")
plt.legend(loc='lower right')
plt.xlabel("k values (for core node)")
plt.ylabel("Precision/Recall")
plt.savefig("avgPosResults.pdf")