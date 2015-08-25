purpose
=======
deepdreamer framework for doing parameterized experiments based on googles deepdream ipython notebook.

requirements
=======
pycaffe.  caffe has its own pile of requirements, so if you can get that to work, you should be able to use this just fine.

example
=======

First, make a folder called 'models', then copy the bvlc_googlenet model data (from caffe) to the models folder, under a directory called "bvlc_googlenet". Alternatively, you can edit the sample_config .json files to point to where you currently have the model stored.

```
python run_dream.py -c sample_configs/simple_demo.json
```

```
python run_dream.py -c sample_configs/guide_demo.json
```

You'll now have deep dreamed images in the folder 'output'.

todo
======
Lots of things.  Like document the full call interface for proc.Proc.process_image.  It's a work in progress.  Don't be too hard on it right now.

license
=======

See LICENSE file