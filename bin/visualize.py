"""Visualization, metrics, and reporting for BasicModel.

Consolidates:
  - Report class (HTML report generation, figure saving, plotting methods)
  - metrics() exploratory space visualizations
  - distances() toy similarity surface plots
"""

import math
import os
from datetime import datetime

import numpy as np
import torch
import util
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

try:
    from torchviz import make_dot
except ImportError:
    make_dot = None

from util import ProjectPaths, TheDevice

OUTPUT_DIR = ProjectPaths.OUTPUT_DIR

def _output_path(filename):
    return ProjectPaths.output_path(filename)

def _output_stem(stem):
    return ProjectPaths.output_stem(stem)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class Report:
    """Collects timestamped SVG figures and XML configs, then writes an HTML report.

    Default ``enabled`` follows ``TheDevice.optimized()`` -- accelerator
    runs (CUDA / MPS) disable reporting by default to keep training
    fast; CPU debug runs auto-enable. Set ``enabled = False`` to
    suppress all figure generation and report output.
    """
    def __init__(self):
        """Build an empty report; pick a timestamp + default enabled flag."""
        self.enabled = not TheDevice.optimized()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.figures = []       # list of (title, svg_path)
        self.xml_configs = []   # list of (name, xml_content)

    def show_figure(self, fig=None):
        """Display figures only on interactive backends; otherwise close them.

        Closes the passed figure on non-interactive backends (Agg /
        template) to avoid leaking matplotlib state. Non-blocking
        show otherwise so the loop continues.
        """
        if not self.enabled:
            if fig is not None:
                plt.close(fig)
            return
        backend = str(plt.get_backend()).lower()
        if "agg" in backend or backend == "template":
            if fig is not None:
                plt.close(fig)
            return
        plt.show(block=False)

    def save_figure(self, fig, title):
        """Save a matplotlib figure as a timestamped SVG and register it.

        Filename slug is ``{timestamp}_{safe_title}.svg``. Appends a
        ``(title, filename)`` pair to ``self.figures`` so the HTML
        writer can embed it later. No-op when ``enabled`` is False.
        """
        if not self.enabled:
            plt.close(fig)
            return None
        safe = title.replace(" ", "_").replace("/", "-")
        filename = f"{self.timestamp}_{safe}.svg"
        path = _output_path(filename)
        fig.savefig(path, format='svg', bbox_inches='tight')
        self.figures.append((title, filename))
        return path

    def add_xml(self, config_path):
        """Register an XML config file for inclusion in the report.

        Reads the file contents into memory; the basename is used as
        the section heading in the rendered HTML.
        """
        if not self.enabled:
            return
        name = os.path.basename(config_path)
        with open(config_path, 'r') as f:
            self.xml_configs.append((name, f.read()))

    def write_html(self):
        """Write the collected figures and configs into a single HTML file.

        Returns ``None`` when reporting is disabled or there's nothing
        to render. Prints the saved file's URL and triggers a VS Code
        open (best-effort) when content was written.
        """
        if not self.enabled:
            return None
        has_tables = hasattr(self, 'tables') and self.tables
        if not self.figures and not has_tables and not self.xml_configs:
            return None
        html_path = _output_path(f"{self.timestamp}_report.html")
        lines = [
            '<!DOCTYPE html>',
            '<html><head>',
            f'<title>BasicModel Report {self.timestamp}</title>',
            '<style>',
            '  body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }',
            '  h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }',
            '  h2 { margin-top: 2em; color: #444; }',
            '  .figure { margin: 1.5em 0; }',
            '  .figure img { max-width: 100%; border: 1px solid #ddd; }',
            '  pre { background: #f5f5f5; padding: 1em; overflow-x: auto; border-radius: 4px; }',
            '  table { border-collapse: collapse; margin: 1em 0; width: 100%; }',
            '  th, td { border: 1px solid #ddd; padding: 0.5em 1em; text-align: left; }',
            '  th { background: #f0f0f0; }',
            '  tr:nth-child(even) { background: #fafafa; }',
            '  .match { color: green; font-weight: bold; }',
            '  .mismatch { color: red; font-weight: bold; }',
            '  .meta { color: #888; font-size: 0.9em; }',
            '</style>',
            '</head><body>',
            f'<h1>BasicModel Report</h1>',
            f'<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]
        # Figures
        lines.append('<h2>Figures</h2>')
        for title, svg_file in self.figures:
            lines.append(f'<div class="figure"><h3>{title}</h3>')
            lines.append(f'<img src="{svg_file}" alt="{title}"></div>')
        # Tables
        if hasattr(self, 'tables') and self.tables:
            for table_title, headers, rows in self.tables:
                lines.append(f'<h2>{table_title}</h2>')
                lines.append('<table>')
                lines.append('<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
                for row in rows:
                    lines.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
                lines.append('</table>')
        # XML configs
        if self.xml_configs:
            lines.append('<h2>Configurations</h2>')
            for name, content in self.xml_configs:
                escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                lines.append(f'<h3>{name}</h3><pre>{escaped}</pre>')
        lines.append('</body></html>')
        with open(html_path, 'w') as f:
            f.write('\n'.join(lines))
        from urllib.parse import quote
        file_url = "file://" + quote(os.path.abspath(html_path))
        print(f"Report saved to {file_url}")
        self._open_in_vscode(html_path)
        return html_path

    def add_table(self, title, headers, rows):
        """Register a table for inclusion in the report.

        Args:
            title: section heading for the table.
            headers: list of column header strings.
            rows: list of lists (one per row).
        """
        if not self.enabled:
            return
        if not hasattr(self, 'tables'):
            self.tables = []
        self.tables.append((title, headers, rows))

    def classificationReport(self, model, min_value=0, max_value=1, **kwargs):
        """Run a model test pass and print a sklearn classification report.

        Saturates predictions into ``[min_value, max_value]`` before
        rounding. Accepts legacy ``min`` / ``max`` kwargs as aliases;
        any other unknown kwarg raises ``TypeError``.
        """
        if not self.enabled:
            return
        if "min" in kwargs:
            min_value = kwargs.pop("min")
        if "max" in kwargs:
            max_value = kwargs.pop("max")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        test_input, _ = model.inputSpace.getTestData()
        if test_input is None:
            raise RuntimeError("classificationReport() requires test input data.")

        batch_size = len(test_input) if hasattr(test_input, "__len__") else 10
        _, _, y_pred, _ = model.runEpoch(
            optimizer=None,
            batchSize=max(1, batch_size),
            split="test",
        )
        y_actual = model.outputSpace.getTestOutput()
        y_actual = np.array(y_actual.cpu() if hasattr(y_actual, "cpu") else y_actual).squeeze()
        y_pred_sat = np.maximum(
            min_value,
            np.minimum(
                max_value,
                np.round(np.array(y_pred.cpu() if hasattr(y_pred, "cpu") else y_pred)).squeeze(),
            ),
        )
        performance = classification_report(
            y_actual, y_pred_sat,
            target_names=["Negative Review", "Positive Review"],
        )
        print(performance)
        return performance

    @staticmethod
    def _open_in_vscode(html_path):
        """Open the report file in VS Code, falling back to the default browser.

        Tries the ``vscode://file/...`` URL via ``open``; on failure
        falls back to ``webbrowser.open`` with a plain ``file://`` URL.
        Best-effort -- silently swallows OSError.
        """
        import os
        import subprocess
        import webbrowser
        from urllib.parse import quote

        abs_path = os.path.abspath(html_path)
        vscode_url = f"vscode://file{quote(abs_path)}"
        file_url = f"file://{quote(abs_path)}"

        try:
            result = subprocess.run(
                ["open", vscode_url],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return
        except OSError:
            pass

        webbrowser.open(file_url)
    # ----- Plotting methods -----

    def mnistReport(self, model):
        """Run test epoch, compute per-class accuracy, and plot.

        Misnamed for legacy reasons: the function reports general
        classification accuracy (per-class correctness rate), not
        MNIST-specific behaviour.  Designed for tasks where each
        test row has a single integer / one-hot label.

        Skips gracefully for:
          * IR / masked-LM text training (post-2026-05-14 default for
            ``dataType=embedding``) — there's no per-row class label
            to score against; the IR P-tier reconstruction loss is
            the live signal.
          * Empty test split (structural-scaffold configs).
          * Zero-epoch runs (no training happened).
        """
        # Skip for IR-only text models: the "predicted vs actual"
        # contract assumes a per-row class label, but masked-LM
        # training produces per-slot predictions over the perceptual
        # codebook — meaningless to argmax-and-compare against a
        # token-count output tensor.
        if getattr(model, 'model_type', None) == 'embedding':
            return torch.zeros(1)
        try:
            test_input = model.inputSpace.getTestData()[0]
            if test_input is None:
                return torch.zeros(1)
            if isinstance(test_input, torch.Tensor) and test_input.numel() == 0:
                return torch.zeros(1)
            if hasattr(test_input, '__len__') and len(test_input) == 0:
                return torch.zeros(1)
        except (AttributeError, IndexError, TypeError):
            pass
        # Also skip when the model is a structural scaffold (numEpochs=0
        # configs like idempotent.xml that exist only to wire the C-S
        # round-trip primitive for unit tests, not to run end-to-end).
        # Without training the head's reverse path can land on
        # mismatched-shape activations; the report has nothing
        # meaningful to compute either way.
        try:
            from util import TheXMLConfig
            num_epochs = TheXMLConfig.get(
                "architecture.training.numEpochs", default=1)
            if num_epochs is not None and int(num_epochs) == 0:
                return torch.zeros(1)
        except (AttributeError, KeyError, TypeError, ValueError, ImportError):
            pass
        model.set_sigma(0)  # suppress exploration for evaluation
        _, _, y_pred, last_x_pred = model.runEpoch(split="test")
        if not isinstance(y_pred, torch.Tensor) or y_pred.numel() == 0:
            return torch.zeros(1)
        if y_pred.dim() == 1 or y_pred.shape[-1] == 1:
            predicted = (y_pred.squeeze() > 0.5).long()
            actual = (model.outputSpace.getTestOutput().squeeze() > 0.5).long()
        else:
            _, predicted = torch.max(y_pred, 1)
            _, actual = torch.max(model.outputSpace.getTestOutput(), 1)
        # test_output now lives on CPU (list-of-tensors kept off the
        # accelerator so DataLoader workers can pickle slices); align
        # ``actual`` to the prediction device before comparisons.
        actual = actual.to(predicted.device)

        nClasses = int(actual.max().item()) + 1
        if model.certainty:
            fwd_layer = (model.outputSpace.linear1
                         if hasattr(model.outputSpace, 'linear1')
                         else model.outputSpace.forwardLinear)
            norms = torch.linalg.norm(fwd_layer.W, dim=0)
            rCorrect = torch.zeros_like(norms)
        else:
            rCorrect = torch.zeros((nClasses))
        for i in range(nClasses):
            total    = (actual == i).sum().item()
            correct  = (actual==i) & (predicted==actual)
            nCorrect = correct.sum().item()
            rCorrect[i] = nCorrect / total if total > 0 else 0.0
            print(f"Correctly predicted {i}: {rCorrect[i]}")
            if model.certainty:
                print(f"Weight norm: {norms[i]}")

        if model.certainty:
            input_matrix = torch.stack((rCorrect, norms))
            correlation_matrix = torch.corrcoef(input_matrix)
            correlation_value = correlation_matrix[0, 1]
            print(f"Pearson Correlation: {correlation_value}")
            from data import TheData
            self.plotAccuracyAndCertainty(model.name, rCorrect, model.reversible, last_x_pred, TheData.test_output)
        else:
            self.plotAccuracy(model.name, rCorrect)
        return rCorrect

    def plotAccuracy(self, model_name, rCorrect):
        """Plot per-class accuracy.

        Renders a single line plot of accuracy vs. class index and
        registers the figure for the HTML report.
        """
        if not self.enabled:
            return
        nClasses = len(rCorrect)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(nClasses), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy per Class: {model_name}")
        plt.legend()
        plt.grid(True)
        self.save_figure(fig, f"{model_name} Accuracy")
        self.show_figure(fig)

    def plotAccuracyAndCertainty(self, model_name, rCorrect, reversible=False,
                                 last_x_pred=None, test_output=None):
        """Plot per-class accuracy with certainty, and optionally reconstruction images.

        For reversible models, also dumps up to 10 reverse-pass
        reconstructions as 28x28 image plots so the visual quality
        of the inverse can be eyeballed.
        """
        if not self.enabled:
            return
        nClasses = len(rCorrect)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(range(nClasses), rCorrect, label="Error (per Input)", marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy & Certainty")
        plt.title(f"Accuracy and Certainty: {model_name}")
        plt.legend()
        plt.grid(True)
        self.save_figure(fig, f"{model_name} Accuracy")
        self.show_figure(fig)

        if reversible and last_x_pred is not None and test_output is not None:
            for i in range(0, 10):
                fig = plt.figure(figsize=(10, 5))
                j = test_output[-i-1]
                _, num = torch.max(j, axis=0)
                plt.title(f"Reconstruction {num}: {model_name}")
                image = last_x_pred[9-i, :]
                image = np.reshape(image, (28, 28))
                plt.imshow(image)
                self.save_figure(fig, f"{model_name} Reconstruction {num}")
                self.show_figure(fig)

    def plotLoss(self, model_name, trainErr, valErr, testErr):
        """Plots the training, validation, and test losses over time.

        ``trainErr`` / ``testErr`` are 2-tuples of (output-loss series,
        reconstruction-loss series); the reconstruction line is drawn
        only when populated. Training is offset by +2 (epoch 1 is test-only).
        """
        if not self.enabled:
            return
        fig = plt.figure(figsize=(10, 5))

        # Training starts at epoch 2 (epoch 1 is test-only), so offset by +2
        plt.plot(range(2, len(trainErr[0]) + 2), trainErr[0], label="Training Error (Output)", marker='o')
        if len(trainErr) > 1 and trainErr[1]:
            plt.plot(range(2, len(trainErr[1]) + 2), trainErr[1], label="Training Error (Reconstruction)", marker='o')

        if testErr:
            plt.plot(range(1, len(testErr[0]) + 1), testErr[0], label="Test Error (Output)", marker='x')
            if len(testErr) > 1 and testErr[1]:
                plt.plot(range(1, len(testErr[1]) + 1), testErr[1], label="Test Error (Reconstruction)", marker='x')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Error per Epoch: {model_name}")
        plt.legend()
        plt.grid(True)

        self.save_figure(fig, f"{model_name} Error")
        self.show_figure(fig)

    def plotActivations(self, figure=1, percepts=None, concepts=None, symbols=None):
        """Plot per-position activation norms for percepts / concepts / symbols.

        Each non-None tensor is rendered as a single norm-vs-position
        curve sharing a 12x4 inch figure. Reuses ``figure`` index if
        already open so live-update workflows don't accumulate windows.
        """
        if not self.enabled:
            return
        if plt.fignum_exists(figure):
            fig = plt.figure(figure)
            fig.set_size_inches(12, 4, forward=True)
            fig.clf()
        else:
            fig = plt.figure(figure, figsize=(12, 4))

        if percepts is not None:
            p = percepts[-1, :, :].squeeze()
            pAct = torch.norm(p, dim=1).detach().numpy()
            plt.plot(pAct, marker='o', color='r')
            plt.xlabel("Activation")
            plt.ylabel("Percepts")
            plt.title("Perceptual Activation")

        if concepts is not None:
            c = concepts[-1, :, :].squeeze()
            cAct = torch.norm(c, dim=-1).detach().numpy()
            plt.plot(cAct, marker='o', color='b')
            plt.xlabel("Epoch")
            plt.ylabel("Concepts")
            plt.title("Conceptual Activation")

        if symbols is not None:
            s = symbols[-1, :, 0].squeeze()
            sAct = s.detach().numpy()
            plt.plot(sAct, marker='o', color='g')
            plt.xlabel("Epoch")
            plt.ylabel("Symbols")
            plt.title("Symbolic Activations")

        plt.tight_layout()
        self.show_figure(fig)

    def plotSpace(self, model):
        """Visualizes learned weight parameters via PCA projections.

        Reduces ``model.prototypes`` (and downstream space weights) to
        2D via sklearn PCA and scatters them; useful for eyeballing
        codebook geometry after training.
        """
        if not self.enabled:
            return
        perc_weights = model.prototypes.data.cpu().numpy()
        pca = PCA(n_components=2)
        perc_2d = pca.fit_transform(perc_weights)

        fig = plt.figure(figsize=(18, 5))

        plt.subplot(1, 3, 1)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Perceptual Space Prototypes")
        plt.legend()

        perc_mean = perc_2d.mean(axis=0)

        conc_weights = model.conceptual.fc_p.weight.data.cpu().numpy()
        conc_proj = pca.transform(conc_weights)

        plt.subplot(1, 3, 2)
        plt.scatter(perc_2d[:, 0], perc_2d[:, 1], c='blue', label="Perceptual Prototypes")
        x_vals = np.linspace(np.min(perc_2d[:, 0]) - 1, np.max(perc_2d[:, 0]) + 1, 100)
        for i in range(conc_proj.shape[0]):
            n = conc_proj[i]
            if np.abs(n[1]) > 1e-3:
                y_vals = perc_mean[1] - (n[0] / n[1]) * (x_vals - perc_mean[0])
                plt.plot(x_vals, y_vals, '--', label=f"Hyperplane {i + 1}" if i == 0 else None, alpha=0.7)
            else:
                plt.axvline(x=perc_mean[0], linestyle='--', alpha=0.7, label=f"Hyperplane {i + 1}" if i == 0 else None)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Conceptual Hyperplanes")
        plt.legend()

        symb_weights = model.fc_symbolic.weight.data.cpu().numpy()
        symb_norms = np.linalg.norm(symb_weights, axis=1)
        x_symb = np.arange(len(symb_norms))

        plt.subplot(1, 3, 3)
        plt.vlines(x_symb, 0, symb_norms, color='k', alpha=0.7)
        plt.scatter(x_symb, symb_norms, color='red', s=100, zorder=3)
        plt.xlabel("Symbolic Feature Index")
        plt.ylabel("L2 Norm")
        plt.title("Symbolic Weights (Lollipop Plot)")

        plt.tight_layout()
        self.show_figure(fig)

    def plotNetwork(self, model):
        """Uses Torchviz to visualize the computation graph.

        Runs one test sample through the model to capture the autograd
        graph, renders it as a PNG via torchviz, and prints the saved
        path. Requires torchviz to have imported successfully.
        """
        if not self.enabled:
            return
        try:
            from basicmodel.bin.Models import TheData
        except ModuleNotFoundError:
            from Models import TheData
        model.eval()
        output, input, _, _ = model.runTest(TheData.test_input, TheData.test_output)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        graph_path = dot.render(_output_stem(f"graph_{model.name}"))
        print(f"Saved network graph as {graph_path}")

    def plotErrorbars(self, model_name, acc):
        """Add an errorbar series for ``model_name`` to the current figure.

        ``acc`` is a [trials, epochs] array; mean and std along trials
        are plotted as the y-value and error bar. Does not call show /
        save -- caller composes a multi-series figure.
        """
        if not self.enabled:
            return
        x = list(range(1, len(acc[0]) + 1))
        y = np.array(np.mean(acc, axis=0))
        y_err = np.std(acc, axis=0)
        plt.errorbar(x, y, yerr=y_err, fmt='-o', label=model_name, capsize=4)

    def plotErrorbarsFromFile(self, fn):
        """Load a CSV of trial accuracies and add an errorbar series to the current plot.

        Reads ``OUTPUT_DIR/<fn>.csv`` as a 2D array (rows = trials,
        cols = epochs). Useful for stitching together comparison plots
        from previously saved per-model CSVs.
        """
        if not self.enabled:
            return
        acc = np.loadtxt(_output_path(f"{fn}.csv"), delimiter=",")
        y = np.mean(acc, axis=0)
        y_err = np.std(acc, axis=0)
        x = list(range(1, len(y) + 1))
        plt.errorbar(x, y, yerr=y_err, fmt='-o', label=fn, capsize=4)

    def plotComparison(self, models):
        """Plot per-digit accuracy comparison across model variants.

        Args:
            models: list of (name, rCorrect_tensor) tuples
        """
        if not self.enabled:
            return
        digits = list(range(10))
        n = len(models)
        width = 0.8 / n
        fig = plt.figure(figsize=(12, 6))
        for i, (name, rCorrect) in enumerate(models):
            offsets = [d + (i - n/2 + 0.5) * width for d in digits]
            vals = rCorrect.detach().cpu().numpy() if hasattr(rCorrect, 'detach') else rCorrect
            plt.bar(offsets, vals, width, label=name)
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.title("Per-Digit Accuracy Comparison")
        plt.xticks(digits)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        path = self.save_figure(fig, "Digit Comparison")
        self.show_figure(fig)
        print(f"Comparison saved to {path}")

    def plotCombinedLoss(self, models):
        """Overlay training and test loss curves from multiple models on shared axes.

        Args:
            models: list of model instances with .trainLosses and .testLosses.
        """
        if not self.enabled:
            return
        fig = plt.figure(figsize=(12, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for i, m in enumerate(models):
            color = colors[i % len(colors)]
            if hasattr(m, 'trainLosses') and m.trainLosses[0]:
                train = m.trainLosses[0]
                # Training starts at epoch 2 (epoch 1 is test-only)
                plt.plot(range(2, len(train) + 2), train,
                         label=f"{m.name} - Training",
                         marker='o', color=color, linestyle='-')
            if hasattr(m, 'testLosses') and m.testLosses[0]:
                test = m.testLosses[0]
                plt.plot(range(1, len(test) + 1), test,
                         label=f"{m.name} - Test",
                         marker='x', color=color, linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Error per Epoch: Model Comparison")
        plt.legend()
        plt.grid(True)
        self.save_figure(fig, "Combined Error Comparison")
        self.show_figure(fig)

    def plotCombinedAccuracy(self, models):
        """Overlay per-digit accuracy curves from multiple models on shared axes.

        Args:
            models: list of (name, rCorrect_tensor) tuples.
        """
        if not self.enabled:
            return
        fig = plt.figure(figsize=(12, 6))
        digits = list(range(10))
        for name, rCorrect in models:
            vals = rCorrect.detach().cpu().numpy() if hasattr(rCorrect, 'detach') else rCorrect
            plt.plot(digits, vals, label=name, marker='o')
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Digit: Model Comparison")
        plt.xticks(digits)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        self.save_figure(fig, "Combined Accuracy Comparison")
        self.show_figure(fig)

    def plotDecisionBoundary(self, model, X, Y, title="Decision Boundary"):
        """Visualise the learned scalar output over a 2D input plane.

        Args:
            model: callable that accepts a (batch, 2, 1) tensor and returns predictions.
            X: numpy array of training inputs, shape (N, 2) or (N, 2, 1).
            Y: numpy array of training targets, shape (N,) or (N, 1) or (N, 1, 1).
            title: plot title.
        """
        if not self.enabled:
            return
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        grid_tensor = grid_tensor.unsqueeze(2)
        with torch.no_grad():
            Z = model(grid_tensor).reshape(xx.shape)
        Z = Z.squeeze()

        fig = plt.figure(figsize=(7, 6))
        plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap="coolwarm", alpha=0.5)
        plt.colorbar(label="Model Output")

        X_2d = X.squeeze() if X.ndim > 2 else X
        Y_flat = Y.squeeze()
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=Y_flat, edgecolors='k',
                    cmap="coolwarm", s=100)
        plt.xlabel("Input Feature x1")
        plt.ylabel("Input Feature x2")
        plt.title(title)
        plt.grid(True)
        self.save_figure(fig, title)
        self.show_figure(fig)

    def plotEpochComparison(self):
        """Plot epoch-level accuracy comparison from saved CSV files.

        Hardcoded list of model-name stems; each is loaded via
        ``plotErrorbarsFromFile`` if its CSV exists in OUTPUT_DIR.
        Output is saved as a single SVG.
        """
        if not self.enabled:
            return
        fig = plt.figure(figsize=(10, 5))
        for fn in ["SimpleModel", "ErgodicModel", "Ergodic - Normed", "Ergodic - Reversible"]:
            csv_path = _output_path(f"{fn}.csv")
            if os.path.exists(csv_path):
                self.plotErrorbarsFromFile(fn)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Model Comparison")
        plt.legend()
        plt.grid(True)
        self.save_figure(fig, "Model Comparison")
        self.show_figure(fig)


TheReport = Report()


# ---------------------------------------------------------------------------
# Metric space visualizations
# ---------------------------------------------------------------------------

# Global variable for subplot indexing
_subplot_index = 0

def plot_things(things=None):
    """Reset the figure or append a new subplot for the given vectors.

    Called with ``None`` to start a fresh four-panel figure (resets
    ``_subplot_index``). Subsequent calls draw one panel each:
    1D arrays as line plots, 2D arrays as origin-anchored rays.
    """
    global _subplot_index

    if things is None:
        # The first call starts a fresh four-panel figure.
        plt.figure(figsize=(6, 10))
        _subplot_index = 0
        return

    _subplot_index += 1
    plt.subplot(4, 1, _subplot_index)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    if things.ndim == 1 or things.shape[1] == 1:
        plt.plot(things)
    else:
        # Multi-dimensional inputs are drawn as rays so orientation is visible.
        for vec in things:
            plt.plot([0, vec[0]], [0, vec[1]])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])


def metrics():
    """Explore several distance metrics in perceptual and conceptual spaces.

    Generates synthetic physical / perceptual / conceptual / symbolic
    rays and renders them in a stacked four-panel figure so the
    transforms between spaces are visible side by side.
    """
    plot_things()  # Initialize figure
    dim = 2
    k = 5

    # === Physical Space ===
    x0 = np.zeros((k, 2))  # Not used further
    objects = 10 * np.random.randn(k, 2)
    plot_things(objects)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    # === Perceptual Space ===
    percepts = objects.copy()
    percepts -= np.min(percepts)
    percepts /= np.max(np.linalg.norm(percepts, axis=1))
    plot_things(percepts)

    # === Conceptual Space ===
    nC = 42
    knowing = 2 * np.random.rand(nC, 2) - 1
    knowing /= np.linalg.norm(knowing, axis=1, keepdims=True)
    activation = percepts @ knowing.T
    activation = activation.sum(axis=0) / k
    activation -= np.mean(activation)
    concepts = (activation[:, np.newaxis] * knowing)
    plot_things(concepts)

    # === Symbolic Space ===
    symbols = np.sign(activation[:, np.newaxis])
    plot_things(symbols)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Distance surface visualizations
# ---------------------------------------------------------------------------

def distances():
    """Plot three toy similarity/distance surfaces over a 2D grid.

    Renders cosine-like, Euclidean, and pi (norm-normalised) surfaces
    over a 2D mesh. Useful for visual debugging of which metric the
    model is implicitly learning at the symbolic head.
    """
    def cos_distance(w, X, Y):
        """Inner-product surface (cosine numerator) parameterized by ``w``."""
        norms = np.sqrt(X ** 2 + Y ** 2) + 1e-8
        dot = w[0] * Y + w[1] * X
        return dot

    def euclid_distance(w, X, Y):
        """Euclidean distance from ``w`` to each grid point ``(X, Y)``."""
        return np.sqrt((X - w[1]) ** 2 + (Y - w[0]) ** 2)

    def pi_distance(w, X, Y):
        """Soft pi-like surface using tanh-saturated axis contributions."""
        return (1 + np.tanh(w[0] * Y)) * (1 + np.tanh(w[1] * X))

    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    w = np.array([0.6, 0.8])
    w = w / np.linalg.norm(w)

    Z1 = cos_distance(w, X, Y)
    Z2 = euclid_distance(w, X, Y)
    Z3 = pi_distance(w, X, Y)

    plot_surface_3d(X, Y, Z1, 'cosDistance (cosine similarity)')
    plot_surface_3d(X, Y, Z2, 'euclidDistance')
    plot_surface_3d(X, Y, Z3, 'piDistance')

    plt.show()


def plot_surface_3d(X, Y, Z, title):
    """Render one surface in its own 3D figure for side-by-side comparison.

    Each call creates a new matplotlib figure so multiple surfaces can
    be inspected together rather than overlaid.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="visualize.py",
        description=(
            "Visualize BasicModel metrics and distance functions.\n\n"
            "Modes:\n"
            "  metrics    Plot training accuracy and loss curves (default).\n"
            "  distances  Plot 3D surface plots of cosine, Euclidean, and\n"
            "             pi-distance functions.\n\n"
            "Examples:\n"
            "  python visualize.py\n"
            "  python visualize.py metrics\n"
            "  python visualize.py distances\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="metrics",
        choices=["metrics", "distances"],
        help="Visualization mode: 'metrics' (default) or 'distances'.",
    )
    args = parser.parse_args()

    if args.mode == "distances":
        distances()
    else:
        metrics()
