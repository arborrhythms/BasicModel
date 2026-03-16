# TODO

## Add a BackicModel.xml
* Create a BasicModel.xml that can be trained by "make basic_train" after creation of the embedding space.
* That model should be relatively large: we want it to be a chat bot. maybe start with 10,000 parameters.
* Ensure that serve.py can function as an interface to that model, passing context and input query to the input, and returning the output as a response.
