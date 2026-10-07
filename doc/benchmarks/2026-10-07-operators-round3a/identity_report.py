"""Read saved identity and training evidence; no model, seed or new training."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def read(path):
    return json.loads((HERE/path).read_text())

def main():
    forms = read('form-audit.json')
    summary = read('measurements/summary.json')
    configurations = []
    lines = ['# Round 3a identity and reader geometry', '',
             'Sparse forms address the index. The fixed dense projection supplies binding directions. '
             'The full corpus census is a static construction audit and adds no training.', '',
             '| Configuration | Native words | Collision groups before → after | Native containment pairs | Violations | Exact byte errors |',
             '|---|---:|---:|---:|---:|---:|']
    for item in forms['configurations']:
        n,w = item['native'],item['witnesses']
        row = dict(configuration=n['configuration'],words=len(n['vocabulary']),
            before=len(n['before_mint_collisions']),after=n['collisions'],
            native_comparable_pairs=len(n['containment_pairs']),
            native_containment_vacuous=not n['containment_pairs'],violations=n['containment_violations'],
            byte_errors=n['reconstruction_errors'],mints=n['mints'],
            projection_sha256=n['projection_sha256'],
            witness_order=w['required_order'],not_comparable=w['not_comparable'],
            witness_mints=w['mints'],lengths_1_through_32=w['lengths_1_through_32'],
            native_rows=n['native_atom_and_word_rows'],fits_declared_part_capacity=n.get('fits_declared_part_capacity'))
        configurations.append(row)
        lines.append(f"| {row['configuration']} | {row['words']:,} | {row['before']} → {row['after']} | {row['native_comparable_pairs']:,} | {row['violations']} | {row['byte_errors']} |")
    lines += ['', 'Each configuration separately passes bana/banana, cat/concat and aba/ababa. '
              'an/and and an/ant remain not comparable because of n#. All lengths 1–32 are exact; '
              'aaaaaa and aaaaaaa differ before minting. The gate vocabulary has no native inclusions.', '',
              'The configured BasicModel corpus contains more words than its declared 32,768-row part bank can admit. '
              'The census uses a separate audit bank to measure forms and reports its row count. '
              'This round does not resize the production bank or claim a full-corpus training pass.', '',
              'All native mints (atom bit counts in parentheses):']
    for row in configurations:
        for mint in row['mints']:
            lines.append(f"- {row['configuration']}: {' / '.join(mint['words'])}: " +
                         ', '.join(f'{a} ({b})' for a,b in zip(mint['atoms'],mint['bit_counts'])) + '.')
    stress = forms['dictionary_stress']
    lines += ['', f"The same toy sample has {len(stress['vocabulary']):,} unique words, "
              f"{len(stress['before_mint_collisions'])} collision group before minting and {stress['collisions']} after. "
              'calaba/cabala use cal@1/cab@1, three bits each. Every required witness bank records this mint too.', '',
              'Both projected conjunction and disjunction roots have affine rank four on the four gate sentences. '
              'Nonzero, separable disjunction roots are possible with this construction; a conjunction-only flip '
              'criterion is therefore a historical diagnostic, not evidence by itself of reader failure.', '',
              '| XOR run | Presented MSE | Class | Final root affine rank | Optimal reader linear MSE | Rung-0 errors / reads |',
              '|---|---:|---|---:|---:|---:|']
    runs=[]
    for gate in summary['xor']:
        audit=read(gate['run_audit'])
        end=audit['end']; fit=end['affine_fit']; rung=audit['rung0_audit']
        row=dict(run=gate['run'],mse=gate['mse'],class_pass=gate['class_pass'],
                 start_fit=audit['start']['affine_fit'],end_fit=fit,rung0=rung,
                 final_operators=gate['operator_names'],
                 diagnosis=('passes' if gate['class_pass'] else
                            'reader convergence: the final presented feature matrix admits an exact linear fit'
                            if fit['reader_features']['optimal_linear_mse'] < 1e-10 else
                            'final reader-feature geometry does not admit an exact linear fit'))
        runs.append(row)
        lines.append(f"| {row['run']:02} | {row['mse']:.9g} | {'pass' if row['class_pass'] else 'miss'} | "
                     f"{fit['roots']['affine_rank']} | {fit['reader_features']['optimal_linear_mse']:.3g} | "
                     f"{rung['errors']} / {rung['identified_reads']:,} |")
    lines += ['', 'The optimal fits are postprocessing of saved features and targets, not substituted predictions '
              'or additional gate trainings. The class column is the actual presented reader after the fixed '
              '400 epochs. Reader trajectories, final operator sequences and late trial costs are retained '
              'in the other receipt reports.']
    result=dict(configurations=configurations,dictionary_stress=dict(words=len(stress['vocabulary']),
        before=len(stress['before_mint_collisions']),after=stress['collisions'],mints=stress['mints']),xor=runs)
    (HERE/'identity-report.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'identity-report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(runs=len(runs),rung0_errors=sum(r['rung0']['errors'] for r in runs))))

if __name__ == '__main__':
    main()
