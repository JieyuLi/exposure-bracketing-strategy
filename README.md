# exposure-bracketing-strategy
A lightweight neural network for exposure bracketing strategy for HDR imaging.

## Download
NightHDR Dataset (https://drive.google.com/drive/folders/1NaO6TdQtWVzBtS_qUqRsjLughrq4UMyq?usp=sharing).
Please download MIT-Adobe FiveK Dataset (https://data.csail.mit.edu/graphics/fivek/) for day scene images, and run FiveKdata.py to prepocess the data.
## Usage
### Requirements
Python3, requirements.txt
### Quick Demo
        python test.py ./ pretrained_model/policy_post_model_best.path.tar --test-list demo_list.txt --results results --score-path results/score.txt
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
