import torch
from models.base import BaseLightningModule
from models.geometric_deeponet.network import GeoDeepONetTime as _GeoDeepONetTime

class GeometricDeepONetTime(BaseLightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # save all hyperparameters (including height, width, lr, etc.)
        self.save_hyperparameters()

        # compute effective channels
        eff = (
            self.hparams.output_channels
            if self.hparams.includePressure
            else self.hparams.output_channels - 1
        )

        # build network args
        net_args = {k: getattr(self.hparams, k) for k in [
            'height', 'width', 'num_input_timesteps', 'input_channels_loc',
            'modes', 'branch_stage1_layers', 'trunk_stage1_layers',
            'branch_stage2_layers', 'trunk_stage2_layers',
            'cnn_c1', 'cnn_c2', 'cnn_c3', 'cnn_fc1', 'cnn_fc2'
        ]}
        net_args['effective_output_channels'] = eff
        self.model = _GeoDeepONetTime(**net_args)
        
    def forward(self, inputs: tuple):
        """
        This is the LightningModule entry point for inference.
        It simply calls through to the underlying torch.nn.Module.
        """
        return self.model(inputs)        