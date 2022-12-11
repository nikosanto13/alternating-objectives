from utils.losses import ce_loss, cw_loss, dlr_loss, alternate_loss, linear_loss
from utils.lr_schedules import a3_schedule, gama_schedule, md_schedule, aa_schedule

# experiments configurations
# declared here because the loss_fn values are functions from utils.losses
loss_configs = {
    'CE': {'loss_fn': ce_loss, 'kwargs': {}, 'name': 'CE'},
    'CW': {'loss_fn': cw_loss, 'kwargs': {}, 'name': 'CW'},
    'DLR': {'loss_fn': dlr_loss, 'kwargs': {}, 'name': 'DLR'},

    'CECW': {'loss_fn': alternate_loss, 'kwargs': {'losses': [ce_loss, cw_loss], 'timesteps': [50, 100]}, 'name': 'Alt-CECW'},
    'CWCE': {'loss_fn': alternate_loss, 'kwargs': {'losses': [cw_loss, ce_loss], 'timesteps': [50, 100]}, 'name': 'Alt-CWCE'},

    'CEDLR': {'loss_fn': alternate_loss, 'kwargs': {'losses': [ce_loss, dlr_loss], 'timesteps': [50, 100]}, 'name': 'Alt-CEDLR'},
    'DLRCE': {'loss_fn': alternate_loss, 'kwargs': {'losses': [dlr_loss, ce_loss], 'timesteps': [50, 100]}, 'name': 'Alt-DLRCE'},

    'CWDLR': {'loss_fn': alternate_loss, 'kwargs': {'losses': [cw_loss, dlr_loss], 'timesteps': [50, 100]}, 'name': 'Alt-CWDLR'},
    'DLRCW': {'loss_fn': alternate_loss, 'kwargs': {'losses': [dlr_loss, cw_loss], 'timesteps': [50, 100]}, 'name': 'Alt-DLRCW'},

    'CECWDLR': {'loss_fn': alternate_loss, 'kwargs': {'losses': [ce_loss, cw_loss, dlr_loss], 'timesteps': [33, 66, 100]}, 'name': 'Alt-CECWDLR'},
    'CEDLRCW': {'loss_fn': alternate_loss, 'kwargs': {'losses': [ce_loss, dlr_loss, cw_loss], 'timesteps': [33, 66, 100]}, 'name': 'Alt-CEDLRCW'},

    'Conv1_CECW': {'loss_fn': linear_loss, 'kwargs': {'losses': [ce_loss, cw_loss], 'coeffs': [0.25, 0.75]}, 'name': 'Conv-CECW(0.25/0.75)'},
    'Conv2_CECW': {'loss_fn': linear_loss, 'kwargs': {'losses': [ce_loss, cw_loss], 'coeffs': [0.75, 0.25]}, 'name': 'Conv-CECW(0.75/0.25)'}
}

# dictionary that maps arguments to step size schedule functions
step_configs = {
    'None': None,
    'A3': a3_schedule,
    'GAMA-PGD': gama_schedule,
    'MD-Attack': md_schedule,
    'AutoAttack': aa_schedule
}
