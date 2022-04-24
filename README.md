# exposure-bracketing-strategy
A lightweight neural network for exposure bracketing strategy for HDR imaging.

## Download
NightHDR Dataset (https://github.com/JieyuLi/night-hdr-dataset).
Please download MIT-Adobe FiveK Dataset (https://data.csail.mit.edu/graphics/fivek/) for day scene images, and run FiveKdata.py to prepocess the data.
## Usage
### Requirements
Python3, requirements.txt
### Evaluation
        python test.py [-h] [--test-list] [--results PATH] [--score-path PATH]
                        DIR DIR
        positional arguments:
        DIR                       path to testset
        DIR                       path to models

        optional arguments:
        -h, --help                show this help message and exit
        --test-list               path to test list(.txt)
        --results                 path to save results
        --score-path              path to save predicted exposure settings and psnr
        
### Training
TODO
