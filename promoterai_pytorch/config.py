from transformers import PretrainedConfig

class PromoterAIConfig(PretrainedConfig):
    model_type = "promoterai"

    def __init__(
        self,
        num_blocks=24,
        model_dim=1024,
        output_dims=[1],
        kernel_size=5,
        shortcut_layer_freq=4,
        output_crop=0,
        input_length=20480,
        output_length=4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.model_dim = model_dim
        self.output_dims = output_dims
        self.kernel_size = kernel_size
        self.shortcut_layer_freq = shortcut_layer_freq
        self.output_crop = output_crop
        self.input_length = input_length
        self.output_length = output_length

        # Optional dilation rate override
        self.dilation_rate = kwargs.get(
            "dilation_rate", lambda x: max(1, 2 ** (x // 2 - 1))
        )
