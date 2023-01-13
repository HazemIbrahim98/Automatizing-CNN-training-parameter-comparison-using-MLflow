class ModelHyperparameters:
    def __init__(self, model_type, number_of_epocs, batch_size, learning_rate, optimizer, loss, metrics):
        self.model_type = model_type #"VGG"
        self.number_of_epocs = int(number_of_epocs) #10 TODO: support multiple values
        self.batch_size = int(batch_size) #10 TODO: support multiple values
        self.learning_rate = float(learning_rate) #0.001 TODO: support multiple values
        self.optimizer = optimizer #"adam"
        self.loss = loss #"categorical_crossentropy"
        self.metrics = metrics #"accuracy"
