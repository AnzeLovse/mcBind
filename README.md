## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

First you should clone into this directory.

```
git clone git@github.com:AnzeLovse/RNAp-learn.git
```

### Installing
Build a Docker image from Dockerfile with the following command: 

```
docker build -t rnap .
```

After you have succsessfully built the image you can run it with: 

```
docker run -ti --device "device":/dev/nvidia0 -v "pwd":/rnap -p 127.0.0.1:8888:8888 -name rnap rnap 

example: docker run -ti --device /dev/nvidia0:/dev/nvidia0 -v /home/alovse/nfs/RNAp-learn:/rnap -p 127.0.0.1:8888:8888 -name rnap rnap 
```

where device "device" is the path to your GPU and "pwd" is the path to the directory containing the notebooks

## Running the tests

Inside the container run jupyter notebook and then connect to ```localhost:8889``` in your browser

```
example run command: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8889
```