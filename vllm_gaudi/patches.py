from vllm.config.vllm import VllmConfig

def patched_get_quantization_config(
        model_config, load_config
    ):
        """Get the quantization config."""
        from vllm.platforms import current_platform

        if model_config.quantization == "mxfp4":
            print(
                "MxFP4Quantization is not supported on HPU. Ignoring the quantization config."
            )
            return None

        if model_config.quantization is not None:
            from vllm.model_executor.model_loader.weight_utils import get_quant_config

            quant_config = get_quant_config(model_config, load_config)
            capability_tuple = current_platform.get_device_capability()

            if capability_tuple is not None:
                capability = capability_tuple.to_int()
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. Minimum "
                        f"capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
            supported_dtypes = quant_config.get_supported_act_dtypes()
            if model_config.dtype not in supported_dtypes:
                raise ValueError(
                    f"{model_config.dtype} is not supported for quantization "
                    f"method {model_config.quantization}. Supported dtypes: "
                    f"{supported_dtypes}"
                )
            quant_config.maybe_update_config(model_config.model)
            return quant_config
        return None

VllmConfig._get_quantization_config = staticmethod(patched_get_quantization_config)