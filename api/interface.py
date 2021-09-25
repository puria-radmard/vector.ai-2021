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
        self.maintained_threads = TopicDictionary()
        if config.provider in ["google_pubsub"]:
            self.subscriber = SubscriberClient()

    def create_topic(self, topic_name):
        super(Consumer, self).create_topic(topic_name)
        if topic_name not in self.maintained_threads:
            self.maintained_threads[topic_name] = {}

    def add_topic(self, topic_name):
        super().add_topic(topic_name)
        if topic_name not in self.maintained_threads:
            self.maintained_threads[topic_name] = {}

    def start_stream(self, topic_name, callback_function, stream_name):
        assert stream_name != "listening", "Stream name cannot be listening"
        assert (
            stream_name not in self.maintained_threads[topic_name]
        ), f"Stream {stream_name} already maintained by {topic_name}"
        engine = generate_consumer(self.config, topic_name)
        self.maintained_threads[topic_name][stream_name] = StoppableThread(
            target=direct_messages,
            args=(self.config, engine, callback_function),
        )
        self.maintained_threads[topic_name][stream_name].start()
        print(f"Started streaming messages from {topic_name} to {callback_function}")

    def kill_stream(self, topic_name, stream_name):
        """Stops collecting messages in self.message_bank[topic_name]
        AFTER next message received"""
        thread = self.maintained_threads[topic_name][stream_name]
        if not thread:
            print(f"No message bank with name {topic_name}")
        elif thread.stopped():
            print(f"Already stopped collecting in {topic_name}")
        else:
            thread.stop()
            del self.maintained_threads[topic_name][stream_name]
            self_name = f"{self=}".split("=")[0]
            print(f"Stopped collecting in {self_name}.message_bank[{topic_name}]")

    def start_listening(self, topic_name):
        engine = generate_consumer(self.config, topic_name)
        self.message_bank[topic_name] = []
        self.maintained_threads[topic_name]["listening"] = StoppableThread(
            target=collect_messages,
            args=(self.config, engine, self.message_bank[topic_name]),
        )
        self.maintained_threads[topic_name]["listening"].start()
        print(f"Started collecting in self.message_bank[{topic_name}]")

    def stop_listening(self, topic_name):
        self.kill_stream(topic_name, stream_name="listening")

    def get_next_message(self, topic_name, max_time=None):
        engine = generate_consumer(self.config, topic_name, max_time)
        return get_next_message(self.config, engine)
