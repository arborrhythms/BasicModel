"""Plot the saved decoder audit; no model construction or training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent/'review12-measurements'
data=json.loads((HERE/'audit-summary.json').read_text())
fig, axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
for binary,color in [(0,'#315caa'),(1,'#b55a2b')]:
    rows=[row for row in data['epochs'] if row['binary']==binary]
    epochs=[row['epoch']+1 for row in rows]
    axes[0].plot(epochs,[row['margin']['mean'] for row in rows],color=color,label=f"{rows[0]['rule_name']} (rule {rows[0]['rule_id']})")
    axes[0].fill_between(epochs,[row['margin']['min'] for row in rows],
        [row['margin']['max'] for row in rows],color=color,alpha=.12)
    axes[1].plot(epochs,[row['gradient_difference']['mean'] for row in rows],color=color,lw=.8)
    axes[2].plot(epochs,[row['fixed_parent_change']['mean'] for row in rows],color=color,lw=.8)
for ax in axes:
    ax.axhline(0,color='#444444',lw=.7,ls='--')
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('STOP − undo logit margin')
axes[0].set_title('Audited XOR run 10: raw logits under the eligibility mask')
axes[0].legend(loc='lower left',frameon=False)
axes[1].set_ylabel('Mean dL/dSTOP − dL/dundo')
axes[2].set_ylabel('Mean margin change per update\n(parent held fixed)')
axes[2].set_xlabel('Training epoch (unchanged 400-epoch budget)')
fig.suptitle('Saved observations only • bands show the range across rows and compose trials',fontsize=11)
fig.savefig(HERE/'decoder-margin.png',dpi=160)
fig.savefig(HERE/'decoder-margin.svg')
