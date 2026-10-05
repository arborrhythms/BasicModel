import json, os
from pathlib import Path
import torch


def test_native_reference_reader(eager_reading):
 from test_mm_xor import _fresh_model
 config=os.environ['ATTENTION_REFERENCE_CONFIG']
 model,_,_=_fresh_model(config)
 model.train()
 with torch.no_grad():model.answer_attention.consume_gate.fill_(.5)
 value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
 _,_,output,_=model.forward(value)
 assert torch.isfinite(output).all()
 reader=model.answer_attention
 named=list(reader.named_parameters())
 gradients=torch.autograd.grad(output.square().mean(),[p for _,p in named],allow_unused=True)
 report={name:None if grad is None else float(grad.abs().sum()) for (name,_),grad in zip(named,gradients)}
 assert report['consume_gate'] is not None and report['consume_gate']>0,report
 assert any(value is not None and value>0 for name,value in report.items() if name.startswith('scorer.')),report
 table=model._attention_words.table
 assert (table.spent<=model.attention_budget).all()
 Path(os.environ['ATTENTION_REFERENCE_OUTPUT']).write_text(json.dumps(dict(config=config,
  source_geometry='unchanged shipped XML',seed=None,output_shape=list(output.shape),
  spent=table.spent.tolist(),budget=model.attention_budget,reader_gradients=report),indent=2)+'\n')
