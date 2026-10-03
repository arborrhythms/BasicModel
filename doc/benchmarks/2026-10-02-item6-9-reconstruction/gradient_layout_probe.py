"""Name every dense cotangent sent to the row-local optimizer."""
def pytest_configure(config):
    import Models, torch
    original=Models.BaseModel.getOptimizer
    def optimizer(model,*a,**kw):
        result=original(model,*a,**kw)
        step=result.step
        def observed(*args,**kwargs):
            for name,p in model.named_parameters():
                if p.grad is not None and p.ndim == 2:
                    print('DICTIONARY-GRAD',name,list(p.shape),str(p.grad.layout),float(p.grad.norm()))
            return step(*args,**kwargs)
        result.step=observed
        return result
    Models.BaseModel.getOptimizer=optimizer
