# FluxYOLOv3
An attempt at a minimal implementation of YOLOv3 in Flux based on https://github.com/eriklindernoren/PyTorch-YOLOv3

Conversion guidance/strategy:
https://philtomson.github.io/blog/2018-06-15-translating-pytorch-models-to-flux.jl-part1-rnn/
https://philtomson.github.io/blog/2018-06-20-translating-pytorch-models-to-flux.jl-part2-running-on-gpu/

### TODO

- [x] Tidy up dataset and training folder structures
- [ ] Convert utils
- [ ] Convert model.py
- [ ] Create exportable functions
- [ ] Refactor train and detect examples using exported functions
