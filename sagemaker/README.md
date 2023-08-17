# Amazon SageMaker

The generation model for Pianola is hosted on an Amazon SageMaker serverless inference endpoint. Serverless inference provides on-demand predictions, removing the need for an always-on instance to host the model. However, there is a cold start period when the endpoint is called after a period of inactivity, therefore calls to the model need to account for possible delays of a few dozen seconds.

The SageMaker endpoint is made available for public HTTP calls via the Amazon API Gateway, following the [workflow](https://aws.amazon.com/blogs/machine-learning/creating-a-machine-learning-powered-rest-api-with-amazon-api-gateway-mapping-templates-and-amazon-sagemaker/) recommended on their blog.

## Specifications

The model is created from the pre-built SageMaker image for Pytorch 2.0.0+cpu, which loads the code in inference.py for handling the model, inputs, predictions, and outputs.

