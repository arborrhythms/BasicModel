# TODO

## Add a BackicModel.xml
* Create a BasicModel.xml that can be trained by "make basic_train" after creation of the embedding space.
* That model should be relatively large: we want it to be a chat bot. maybe start with 100,000 parameters, mostly concepts.
* Ensure that basicmodel/bin/serve.py can function as an interface to that model, passing context and input query to the input, and returning the output as a response.

## Add OpenRouter as a provider
* Add openrouter as a provider
* Ensure that basicmodel and nanochat are models for the WikiOracle provider

## Ensure that basicmodel is working from WikiOracle
* Add data/basicmodel.service for WikiOracle.org to route calls to basicmodel/bin/serve.py
