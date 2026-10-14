# Third-party notices

This project contains code adapted from the following open-source projects.

## LLM Foundry (Apache-2.0)

`elastic/eval/icl/datasets.py`, `elastic/eval/icl/utils.py` and `elastic/eval/icl/metrics.py` are
adapted from LLM Foundry v0.22.0 (<https://github.com/mosaicml/llm-foundry>), Copyright 2024
MosaicML LLM Foundry authors, licensed under the Apache License, Version 2.0 (the same license as
this project; see `LICENSE`). The changes are listed at the top of each file. The evaluation task
files fetched by `scripts/download_eval_data.py` are distributed with LLM Foundry (ARC-Easy,
ARC-Challenge, BoolQ and MMLU in OpenLM's copies); each keeps the license of its original dataset.

## Composer (Apache-2.0)

The batch-splitting helpers in `elastic/eval/icl/utils.py` and the evaluation loop in
`elastic/eval/icl/runner.py` follow Composer (<https://github.com/mosaicml/composer>), Copyright
2022 MosaicML Composer authors, licensed under the Apache License, Version 2.0.

## xFormers (BSD-3-Clause)

The rotary position embedding in `elastic/models/layers.py` is adapted from
<https://github.com/facebookresearch/xformers> (as vendored by OpenLM).

```
From xFormers:

Copyright (c) Facebook, Inc. and its affiliates

===

BSD 3-Clause License

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America and IDIAP
   Research Institute nor the names of its contributors may be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## OpenLM (MIT)

Model conventions (rotary embedding usage, pre-tokenized WebDataset text format) follow
<https://github.com/mlfoundations/open_lm>.

```
MIT License

Copyright (c) 2023 mlfoundations

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
