# AI Enterprise Workflow Capstone Project

This project uses time series to forecast revenue prediction for the next 30 days.

# Usage details

Execute commands are from current root directory.

## To test `app.py`

```bash
~$ python app.py
```

or to start the flask app in debug mode

```bash
~$ python app.py -d
```

http://0.0.0.0:8080/ is now available.
    
## For model testing or train

See example of usage in `model.py`

```bash
~$ python model.py
```

## For building docker

```bash
~$ docker build -t ai_capstone .
```

## Run the unittests

Before running the unit tests launch the `app.py`.

To run only the api tests

```bash
~$ python unittests/ApiTests.py
```

To run only the model tests

```bash
~$ python unittests/ModelTests.py
```

To run all of the tests

```bash
~$ python run-tests.py
```

## Run the container to test that it is working  

```bash
~$ docker run -p 4000:8080 ai_capstone
```

Go to http://0.0.0.0:4000/ and you will see a basic website that can be customtized for a project.




