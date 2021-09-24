Part 3 
- Let’s do something real world now! We have multiple machine learning services that are coordinated via a message broker.
- Here, you have to design and build a robust system to classify fashion images. (Here, we can use the fashion MNIST validation set to mock the input) The system will have a single client consuming a single machine learning service.
- Use the model from Part 1 and the library from Part 2 to build such an application. It does have to be robust, scalable and able to process requests asynchronously.
- Note that this is not a REST API based system but rather one which can process requests in a non-blocking way and (theoretically) put the results somewhere else (like a database). You can mock this by printing to the console.