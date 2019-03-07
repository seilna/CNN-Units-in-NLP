predictor_config = {
	"filter_width": 3,
	"dilations": [1, 2, 4, 8, 16,
				  1, 2, 4, 8, 16,
				  1, 2, 4, 8, 16,
				  1, 2, 4, 8, 16,
				  1, 2, 4, 8, 16,
				  ],
	"residual_channels": 512,
	"n_target_quant": 256,
	"n_source_quant": 256,
	"sample_size" : 300
}

translator_config = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}

translator_enc15 = translator_config

translator_enc8 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
                              1, 2, 4
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}


translator_enc6 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
                              1
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}

translator_enc4 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8,
                              
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}

translator_enc2 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}

translator_enc1 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,
}



translator_config_without_layernorm = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": False,
}

translator_config_wide = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 1024,
        "layer_norm": True,

}
translator_config_deep = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
                                                  1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,

}

translator_config_shallow = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
			      1, 2, 4, 8, 16,
			      ],

	"decoder_dilations": [1, 2, 4, 8, 16,
			      1, 2, 4, 8, 16,
			      1, 2, 4, 8, 16,
			    ],
	"residual_channels": 512,
        "layer_norm": True,

}


translator_config_shallowest = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
        "layer_norm": True,

}

translator_config1 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
}

translator_config2 = {
	"decoder_filter_width": 3,
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8,
						  1, 2, 4, 8,
						  ],
	"decoder_dilations": [1, 2, 4, 8,
						  1, 2, 4, 8,
						  ],
	"residual_channels": 512,
}

classifier_config = {
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8,
						  1, 2, 4, 8,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
}

classifier_config2 = {
	"encoder_filter_width" : 5,
	"encoder_dilations": [1, 2, 4, 8,
						  1, 2, 4, 8,
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
}


classifier_config3 = {
	"encoder_filter_width" : 3,
	"encoder_dilations": [1, 2, 4, 8
						  ],
	"decoder_dilations": [1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  1, 2, 4, 8, 16,
						  ],
	"residual_channels": 512,
}
