from .utils import * 
from google.api_core.exceptions import AlreadyExists
from kafka.errors import TopicAlreadyExistsError
from threading import Thread
from google.pubsub_v1 import SubscriberClient

__all__ = ["Producer", "Consumer", "Config"]

class APIItem:
    def __init__(self, config: Config):
        self.config = config
        if self.__class__ == Producer:
            config.producer = self
        elif self.__class__ == Consumer:
            config.consumer = self

    def create_topic(self, topic_name):
        try:
            create_topic(self.config, topic_name, True)
        except (TopicAlreadyExistsError, AlreadyExists) as e:
            print(f"Caught: {e}")
            print("Topic added anyway")

    def add_topic(self, topic_name):
        create_topic(self.config, topic_name, False)


class Producer(APIItem):
    def __init__(self, config: Config):
        super(Producer, self).__init__(config)    
        self.engine = generate_producer(self.config)

    def send_message(self, message, topic_name):
        send_message(message, self.config, topic_name)


class Consumer(APIItem):
    def __init__(self, config: Config):
        super(Consumer, self).__init__(config) 
        self.message_bank = TopicDictionary()
        self.maintained_threads = {} # TopicDictionary()
        self.subscriber = SubscriberClient()

    def start_listening(self, topic_name):
        engine = generate_consumer(self.config, topic_name)
        self.message_bank[topic_name] = []
        self.maintained_threads[topic_name] = StoppableThread(
            target=collect_messages,
            args=(self.config, engine, self.message_bank[topic_name]),
        )
        self.maintained_threads[topic_name].start()
        print(f"Started collecting in self.message_bank[{topic_name}]")

    def stop_listening(self, topic_name):
        """
        Stops collecting messages in self.message_bank[topic_name] AFTER next message received
        """
        thread = self.maintained_threads[topic_name]
        if not thread:
            print(f"No message bank with name {topic_name}")
        elif thread.stopped():
            print(f"Already stopped collecting in {topic_name}")
        else:
            self.maintained_threads[topic_name].stop()
            self_name = f"{self=}".split("=")[0]
            print(f"Stopped collecting in {self_name}.message_bank[{topic_name}]")

    def get_next_message(self, topic_name, max_time = None):
        engine = generate_consumer(self.config, topic_name, max_time)
        return get_next_message(self.config, engine)
