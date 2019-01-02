# FluxYOLOv3
An attempt at a minimal implementation of YOLOv3 in Flux based on https://github.com/eriklindernoren/PyTorch-YOLOv3

Conversion guidance/strategy:
- https://philtomson.github.io/blog/2018-06-15-translating-pytorch-models-to-flux.jl-part1-rnn/
- https://philtomson.github.io/blog/2018-06-20-translating-pytorch-models-to-flux.jl-part2-running-on-gpu/

### TODO

- [x] Tidy up dataset and training folder structures
- [x] Convert dataset download functions to .jl to help generalization
- [ ] Convert utils
- [ ] Convert model.py
- [ ] Create exportable functions
- [ ] Refactor train and detect examples using exported functions


## Downloading a dataset
Datasets are not included in this package due to their size, but can be downloaded using the provided download scripts.

Coco (~21 GB of zip files):

`include("<pkg root>/datasets/coco/get_coco_dataset.jl")`
