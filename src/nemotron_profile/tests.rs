use super::*;

#[test]
fn pinned_nano_profile_keeps_order_geometry_and_routing() {
    let bytes = include_bytes!("../../tests/fixtures/nemotron-nano-config.json");
    let profile = parse_pinned_nano_config(bytes).unwrap();
    let architecture = profile.architecture;
    assert_eq!(profile.max_positions, 262_144);
    assert_eq!(
        (architecture.hidden_size, architecture.num_hidden_layers),
        (2688, 52)
    );
    assert_eq!(
        (
            architecture.num_attention_heads,
            architecture.num_key_value_heads,
            architecture.head_dim
        ),
        (32, 2, 128)
    );
    let hybrid = architecture
        .decoder_extensions
        .mamba2_hybrid
        .as_ref()
        .unwrap();
    assert_eq!(
        (
            hybrid.heads,
            hybrid.head_dim,
            hybrid.groups,
            hybrid.state_dim,
            hybrid.conv_kernel
        ),
        (64, 64, 8, 128, 4)
    );
    assert_eq!(
        (
            hybrid.dt_min,
            hybrid.dt_max,
            hybrid.conv_bias,
            hybrid.linear_bias
        ),
        (0.0, None, true, false)
    );
    assert_eq!(
        hybrid
            .layers
            .iter()
            .filter(|&&block| block == HybridDecoderBlock::Mamba2)
            .count(),
        23
    );
    assert_eq!(
        hybrid
            .layers
            .iter()
            .filter(|&&block| block == HybridDecoderBlock::Attention)
            .count(),
        6
    );
    assert_eq!(
        hybrid
            .layers
            .iter()
            .filter(|&&block| block == HybridDecoderBlock::RoutedMoe)
            .count(),
        23
    );
    assert_eq!(architecture.rope_layers, Some(vec![false; 52]));
    let moe = architecture.mixture_of_experts.unwrap();
    assert_eq!(
        (
            moe.experts,
            moe.experts_per_token,
            moe.expert_intermediate_size,
            moe.shared_expert_intermediate_size
        ),
        (128, 6, 1856, Some(3712))
    );
    let routing = architecture.expert_routing.unwrap();
    assert_eq!(routing.scoring, ExpertScoring::Sigmoid);
    assert_eq!(routing.normalization, Some(ExpertNormalization::Always));
    assert_eq!(routing.routed_scaling_factor, 2.5);
    assert!(routing.selection_bias && !routing.shared_expert_gated);
}

#[test]
fn any_unpinned_configuration_is_rejected() {
    let bytes = include_bytes!("../../tests/fixtures/nemotron-nano-config.json");
    let mut changed = bytes.to_vec();
    changed.push(b' ');
    assert!(matches!(
        parse_pinned_nano_config(&changed),
        Err(ProductionIngestionError::UnsupportedFormat { .. })
    ));
    assert!(matches!(
        parse_pinned_nano_config(b"{}"),
        Err(ProductionIngestionError::UnsupportedFormat { .. })
    ));
}
