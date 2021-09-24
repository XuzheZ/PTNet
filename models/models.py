def create_model(opt):
    if opt.model == 'local':
        from .PTN_model import PTN_local
        model = PTN_local()
    else:
        from .PTN_model import PTN
        model = PTN()

    return model
