# Pinned Olala metadata

`model.header` is the exact 103,288-byte Safetensors header (including the
length prefix) from `OVHaiLLM/Olala-v1.0-mini-7A1B` revision
`b40f8165ee2254c18601220e19842600c6d91794`; the full file is 13,579,677,592
bytes. The header SHA-256 is
`81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5`.

`layer0-geodesic.bin` contains only the six consecutive F32 scalar parameters
and buffers for layer 0's mixer and MLP (24 bytes, immediately after the
header). Its SHA-256 is
`b14815a9aad6ed3d1a74a05e4a717eb8a40c9daa24c77078d7f3ec125706d75e`.
No other model weights are included.

[Pinned checkpoint](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/b40f8165ee2254c18601220e19842600c6d91794/model.safetensors)
