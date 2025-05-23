import torch
from models.base import BaseLightningModule
from models.geometric_deeponet.network import GeoDeepONetTime as _GeoDeepONetTime
from models.deriv_calc import DerivativeCalculator

class GeometricDeepONetTime(BaseLightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        eff = self.hparams.output_channels - 1
        
        self.deriv_calc = DerivativeCalculator(
            height=self.hparams.height,
            width=self.hparams.width,
            domain_length_x=self.hparams.domain_length_x,
            domain_length_y=self.hparams.domain_length_y,
            device=torch.device('cpu'),
            channels=eff
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