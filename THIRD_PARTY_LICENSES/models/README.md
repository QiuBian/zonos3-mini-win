# The model

The add-on downloads Zyphra's ZONOS3-Mini the first time a line is spoken; none
of it is in this archive except the 27 default voices' speaker embeddings
(`bin/_internal/zonos3_addon/voices.json`). Everything below is from
[Zyphra/ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) on Hugging
Face, at the revision `bin/_internal/zonos3_addon/pins.json` names, and is
released by Zyphra, Inc. under the Apache License 2.0
([Apache-2.0.txt](Apache-2.0.txt)).

| File | What it is | When it downloads |
|------|------------|-------------------|
| `gguf/backbone-ptq.gguf` (663 MB) | The language model: Q4_0 experts, Q8_0 mixers, with its tokenizer | The first line |
| `gguf/zac-keep2.gguf` (452 MB) | The waveform decoder | The first line |
| `gguf/spk-encoder.gguf` (45 MB) | The speaker encoder | The first cloned voice |
| `gguf/voices/*/*.embedding.npy` | The default voices' speaker embeddings, carried in `voices.json` | Bundled |
