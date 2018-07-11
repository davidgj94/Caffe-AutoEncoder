WORK_DIR=$(pwd)
cd ..
ROOT_DIR=$(pwd)
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
cd $WORK_DIR

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python solve.py

