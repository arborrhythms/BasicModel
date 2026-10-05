"""Plot saved reader weights; no forward, model, optimizer or RNG draw."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent/'review20-measurements'
s=json.loads((H/'summary.json').read_text())
fig,axes=plt.subplots(2,5,figsize=(14,5),sharex=True,sharey=True,layout='constrained')
for ax,row in zip(axes.flat,s['xor']):
    weights=row['run_audit']['reader_weights']
    ax.plot([w['epoch'] for w in weights],
            [w['parameters']['answer_record_reader.weight'] for w in weights],
            color='#176d53' if row['class_pass'] else '#9b562b',lw=1.4)
    ax.set_title(f"Run {row['run']} · MSE {row['mse']:.4f}",fontsize=10)
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
for ax in axes[1]:ax.set_xlabel('Epoch')
for ax in axes[:,0]:ax.set_ylabel('Reader weight norm')
fig.suptitle('XOR_grammar: affine reader weight norm across the unchanged 400 epochs')
fig.savefig(H/'reader-weight-trajectories.png',dpi=160)
fig.savefig(H/'reader-weight-trajectories.svg')
