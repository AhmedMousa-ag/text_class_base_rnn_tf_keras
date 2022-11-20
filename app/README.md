# Navigate Code

## 1- train

If it's first time to run this docker image, then it's a must to run **train** file to build the model in order to make predictions later.

cmd: docker run -v "($pwd):/opt/ml_vol" --rm `<container image name>` train

## 2- test

To conduct a test for the model.
