import matplotlib.pyplot as plt
import pandas as pd

traditional_path = 'param_tl_results/traditional/traditional_test_loss_m311.csv'
traditional_df = pd.read_csv(traditional_path, index_col=0, header=None)

param_tl_path = 'param_tl_results/param_tl/param_tl_test_loss_m311_final_draft.csv'
param_tl_df = pd.read_csv(param_tl_path, index_col=0, header=None)

# Append 50-len(no.of lines in param_tl_df) to param_tl_df
param_tl_df_init_len = len(param_tl_df.index)
replica_val = param_tl_df.iloc[param_tl_df_init_len-1]

for i in range(50-len(param_tl_df.index)):
    param_tl_df.loc[len(param_tl_df.index) + i] = [replica_val]

ax = plt.subplot(1,1,1)
ax.plot(traditional_df, marker='.')
ax.plot(param_tl_df[:param_tl_df_init_len], marker='.', color='orange')
ax.plot(param_tl_df[param_tl_df_init_len:], color='orange', linestyle = '--')

ax.set_title("Machine-3-11")
ax.set_xlim(-1,50)
# ax.set_ylim(0.004,0.018)
ax.set_xlabel('No.of epochs')
ax.set_ylabel('Test loss')
ax.legend(['Traditional training', 'Similarity-based Parameter Transfer Learning'])
# plt.show()
plt.savefig('final_plots/m34tom311.png')