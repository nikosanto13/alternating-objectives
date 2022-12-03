from utils.losses import CE, CW, DLR, Alternate, Linear
from utils.lr_schedules import A3_Schedule, GAMA_Schedule, MD_Schedule, AA_Schedule

# experiments configurations
# declared here because the loss_fn values are functions from utils.losses
loss_configs = {
    'CE': {'loss_fn': CE, 'kwargs': {}, 'name': 'CE'},
    'CW': {'loss_fn': CW, 'kwargs': {}, 'name': 'CW'},
    'DLR': {'loss_fn': DLR, 'kwargs': {}, 'name': 'DLR'},

    'CECW': {'loss_fn': Alternate, 'kwargs': {'losses': [CE, CW], 'timesteps': [50, 100]}, 'name': 'Alt-CECW'},
    'CWCE': {'loss_fn': Alternate, 'kwargs': {'losses': [CW, CE], 'timesteps': [50, 100]}, 'name': 'Alt-CWCE'},

    'CEDLR': {'loss_fn': Alternate, 'kwargs': {'losses': [CE, DLR], 'timesteps': [50, 100]}, 'name': 'Alt-CEDLR'},
    'DLRCE': {'loss_fn': Alternate, 'kwargs': {'losses': [DLR, CE], 'timesteps': [50, 100]}, 'name': 'Alt-DLRCE'},

    'CWDLR': {'loss_fn': Alternate, 'kwargs': {'losses': [CW, DLR], 'timesteps': [50, 100]}, 'name': 'Alt-CWDLR'},
    'DLRCW': {'loss_fn': Alternate, 'kwargs': {'losses': [DLR, CW], 'timesteps': [50, 100]}, 'name': 'Alt-DLRCW'},

    'CECWDLR': {'loss_fn': Alternate, 'kwargs': {'losses': [CE, CW, DLR], 'timesteps': [33, 66, 100]}, 'name': 'Alt-CECWDLR'},
    'CEDLRCW': {'loss_fn': Alternate, 'kwargs': {'losses': [CE, DLR, CW], 'timesteps': [33, 66, 100]}, 'name': 'Alt-CEDLRCW'},

    'Conv1_CECW': {'loss_fn': Linear, 'kwargs': {'losses': [CE, CW], 'coeffs': [0.25, 0.75]}, 'name': 'Conv-CECW(0.25/0.75)'},
    'Conv2_CECW': {'loss_fn': Linear, 'kwargs': {'losses': [CE, CW], 'coeffs': [0.75, 0.25]}, 'name': 'Conv-CECW(0.75/0.25)'}
}

# dictionary that maps arguments to step size schedule functions
step_configs = {
    'None': None,
    'A3': A3_Schedule,
    'GAMA-PGD': GAMA_Schedule,
    'MD-Attack': MD_Schedule,
    'AutoAttack': AA_Schedule
}
