import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

traditional_path = 'param_tl_results/traditional/traditional_test_loss_m311.csv'
traditional_df = pd.read_csv(traditional_path, index_col=0, header=None)

param_tl_path = 'param_tl_results/param_tl/param_tl_test_loss_m311.csv'
param_tl_df = pd.read_csv(param_tl_path, index_col=0, header=None)

cluster_path = 'param_tl_results/cluster_level_results/cluster_4_test_loss.csv'
cluster_df = pd.read_csv(cluster_path, index_col=0, header=None)

general_path = 'param_tl_results/general_model/general_test_loss.csv'
general_df = pd.read_csv(general_path, index_col=0, header=None)

# Append 50-len(no.of lines in param_tl_df) to param_tl_df
param_tl_df_init_len = len(param_tl_df.index)
replica_val = param_tl_df.iloc[param_tl_df_init_len-1]

for i in range(50-len(param_tl_df.index)):
    param_tl_df.loc[len(param_tl_df.index) + i] = [replica_val]

ax = plt.subplot(1,1,1)
line1, = ax.plot(traditional_df, marker='.', label = 'Traditional training')
line2, =ax.plot(param_tl_df[:param_tl_df_init_len], marker='.', color='orange', label = 'Intra-cluster Parameter\n Transfer Learning')
line3, =ax.plot(param_tl_df[param_tl_df_init_len:], color='orange', linestyle = '--')
line4, =ax.plot(cluster_df, marker='.', color='green', label = 'Cluster-level model training')
line5, =ax.plot(general_df, marker='.', color='black', label = 'General model')
plt.rcParams.update({'font.size': 15})
ax.set_title("Device-3-11")
ax.set_xlim(-1,50)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_ylim(0.004,0.018)
ax.set_xlabel('No.of epochs', fontsize = 16)
ax.set_ylabel('Test loss', fontsize = 16)
ax.legend(handles = [line1,line2,line4, line5])
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
# plt.show()
plt.savefig('tpds_plots/m34tom311.png')