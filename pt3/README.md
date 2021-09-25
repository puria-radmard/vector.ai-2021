Part 3 
- Let’s do something real world now! We have multiple machine learning services that are coordinated via a message broker.
- Here, you have to design and build a robust system to classify fashion images. (Here, we can use the fashion MNIST validation set to mock the input) The system will have a single client consuming a single machine learning service.
- Use the model from Part 1 and the library from Part 2 to build such an application. It does have to be robust, scalable and able to process requests asynchronously.
- Note that this is not a REST API based system but rather one which can process requests in a non-blocking way and (theoretically) put the results somewhere else (like a database). You can mock this by printing to the console.



Implemented process:
- pt3.simulated_client.image_stream sends a stream of images and corresponding indices to topic1, with about 5s between each image (Gaussian delay)
- The service has a callback to all images that it finds in topic1. Upon collection, ...
- A pretrained CNN (cnn.api.load_model) predicts the labels of these images (cnn.api.predict_numpy_images) and sends it to topic2, along with its corresponding indices
- TODO: The client collects these prediction and index pairs
- TODO: After some random delay (average 120s), the client (pt3.simulated_client.###) will send the index and the true label of a received image to topic3 (regardless of whether the service predicted correctly)
- TODO: Similarly to topic1, the service periodically checks topic3. If it finds unseen indices in there, it will accumulate them (and the corresponding images) into its dataset (i.e. including the data it was trained on so far)
- TODO: After a chosen number M images & their labels are added to the dataset, the model will finetune on the dataset so far. This is done in the background so as to not disturb the service
- TODO: After finetuning is done, the path to the model parameters is updated, and the new weights are used in production