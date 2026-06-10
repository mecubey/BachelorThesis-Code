import sys
sys.path.insert(0, '')

from implementation.mapf_utils import EXPERIMENT_RESULTS_DIR
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open(EXPERIMENT_RESULTS_DIR / "random-32-32-10_random_syrup", "rb"))

plt.plot(data["agents"][::5], data["socs"][0][::5], label="HUA")
plt.plot(data["agents"][::5], data["socs"][1][::5], label="HA")
plt.legend()
plt.show()