"""Plot recorded final focus/word extents; no model execution or training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

folder = Path(__file__).resolve().parent
results = json.loads((folder / 'objective-repaired-results.json').read_text())['enabled']
labels = ('One word', 'Two-word requests', 'Sentence', 'Asked word / phrase')
word_colors = {'red': '#a44135', 'blue': '#2d5d91',
               'green': '#326f54', 'gold': '#996912'}
figure, axes = plt.subplots(4, 2, figsize=(13.5, 10.5), constrained_layout=True)
for row, (stage, title) in enumerate(zip(results, labels)):
    measured = stage['final']
    field = measured['trials'][0]
    assert measured['iterations'] == [1, 1, 1, 1]
    selected = field['reads'][0]['regions']
    for axis, name in enumerate(('where', 'when')):
        plot = axes[row, axis]
        scale = measured['coordinate_scale'][axis]
        for batch in range(4):
            y = 3 - batch
            low, high = (x * scale for x in selected[axis][batch][0])
            plot.add_patch(Rectangle((low, y - .23), high - low, .46,
                facecolor='#dcecf8', edgecolor='#467fa8', linewidth=1))
            words = stage['stage']['fields'][batch].split()
            for index, (bounds, word) in enumerate(zip(field[name][batch], words)):
                left, right = (x * scale for x in bounds)
                word_y = y + .11 * (index - (len(words) - 1) / 2)
                plot.plot((left, right), (word_y, word_y),
                          color=word_colors[word], linewidth=3)
        plot.autoscale_view()
        plot.set_ylim(-.5, 3.85)
        plot.set_yticks(range(4), ('row 4', 'row 3', 'row 2', 'row 1'))
        plot.set_title(f'{title} · .{name}', loc='left', fontsize=11)
        plot.ticklabel_format(axis='x', style='plain', useOffset=False)
        plot.grid(axis='x', alpha=.15)
        plot.spines[['top', 'right']].set_visible(False)
        plot.set_xlabel('Native codebook coordinate' if axis == 0 else 'Native occurrence coordinate', fontsize=9)
figure.suptitle('Repaired run: final focus surrounds every observed word\n'
    'Head 0 only · all four batch rows · one field iteration per row', fontsize=15)
figure.legend(handles=[Patch(facecolor='#dcecf8', edgecolor='#467fa8', label='Focus extent'),
    *[Line2D([0], [0], color=color, linewidth=3, label=f'{word} extent')
      for word, color in word_colors.items()]],
    loc='outside lower center', ncol=5, frameon=False)
figure.savefig(folder / 'objective-focus-extents.png', dpi=180)
figure.savefig(folder / 'objective-focus-extents.svg')
