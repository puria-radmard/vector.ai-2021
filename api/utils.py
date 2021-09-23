from datetime import datetime
import json
from google.api_core import retry
from google.cloud import pubsub_v1
import kafka
from typing import *
from .google_utils import * 
from concurrent import futures
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from google.cloud.pubsub_v1 import PublisherClient, SubscriberClient
from kafka.errors import KafkaTimeoutError
import threading
import os

__all__ = [
    "Config",
    "TopicDictionary",
    "StoppableThread",
    "generate_producer",
    "send_message",
    "generate_consumer",
    "get_next_message",
    "collect_messages",
    "create_topic",
]

allowable_providers = ["kafka", "google_pubsub"]
producer_types = Union[KafkaProducer, PublisherClient]
consumer_types = Union[KafkaConsumer, SubscriberClient]
message_types = Union[kafka.consumer.fetcher.ConsumerRecord]


class Config:
    def __init__(self, provider, server=None, project_id=None, credentials_path=None):
        self.provider = provider
        self.server = server
        self.project_id = project_id
        self.topics = TopicDictionary()

        assert (
            provider in allowable_providers
        ), f"Provider must be one of {''.join(a for a in allowable_providers)}"
        
        if self.provider == "kafka":
            assert server != None, "Kafka config requires server"
            self.client = KafkaAdminClient(
                bootstrap_servers=self.server, 
                client_id='test'
            )
        elif self.provider == "google_pubsub":
            (project_id != None and credentials_path != None), \
                "Kafka config requires project_id and credentials path"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def create_topic(config: Config, topic_name: str, create: bool) -> None:
    if config.provider == "kafka":
        config.topics[topic_name] = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
        create and config.client.create_topics([config.topics[topic_name]], validate_only=False)

    elif config.provider == "google_pubsub":
        config.topics[topic_name] = config.producer.engine.topic_path(config.project_id, topic_name)
        create and config.producer.engine.create_topic(name = config.topics[topic_name])


def generate_producer(config: Config) -> producer_types:
    if config.provider == "kafka":
        return KafkaProducer(bootstrap_servers=[config.server], api_version=(0, 10, 1))

    elif config.provider == "google_pubsub":
        return PublisherClient()


def send_message(message: str, config: Config, topic: str) -> None:
    if topic not in config.topics:
        print(f"No such topic {topic}")
        return
    elif config.provider == "kafka":
        config.producer.engine.send(topic, json.dumps(message).encode("utf-8"))

    elif config.provider == "google_pubsub":
        publish_future = config.producer.engine.publish(
            config.topics[topic], message.encode("utf-8")
        )
        publish_future.add_done_callback(get_callback(publish_future, message))
        futures.wait([publish_future], return_when=futures.ALL_COMPLETED)
    print(f"Sent message to {topic}")


def generate_consumer(config: Config, topic_name: str, _time: float = float('inf')):
    if config.provider == "kafka":
        if _time:
            return KafkaConsumer(
                topic_name,
                bootstrap_servers=[config.server],
                api_version=(0, 10),
                consumer_timeout_ms=_time * 1000,
            )
        else:
            return KafkaConsumer(
                topic_name, bootstrap_servers=[config.server], api_version=(0, 10)
            )

    elif config.provider == "google_pubsub":
        subscriber = pubsub_v1.SubscriberClient()
        subscription_id = f"{topic_name}-subscription-{datetime.now().strftime('%S-%M-%H')}"
        subscription_path = subscriber.subscription_path(config.project_id, subscription_id)
        topic_path = generate_producer(config).topic_path(config.project_id, topic_name)
        with subscriber:
            subscription = subscriber.create_subscription(
                request={"name": subscription_path, "topic": topic_path}
            )
        return subscription_path


def get_next_message(config: Config, consumer, timeout: float = 2678400):
    if config.provider == "kafka":
        try:
            for m in consumer:
                return m
        except KafkaTimeoutError as e:
            print(e)

    elif config.provider == "google_pubsub":
        subscriber = SubscriberClient()
        with subscriber:
            response = subscriber.pull(
                request={
                    "subscription": consumer,
                    "max_messages": 1,
                }
            )
            subscriber.acknowledge(
                request={
                    "subscription": consumer,
                    "ack_ids": [msg.ack_id for msg in response.received_messages]
                }
            )
            return [msg for msg in response.received_messages][0]


def collect_messages(config: Config, consumer: consumer_types, bank_list: List) -> None:
    if config.provider == "kafka":
        for m in consumer:
            bank_list.append(m)
            if threading.current_thread().stopped():
                break

    elif config.provider == "google_pubsub":
        subscriber = SubscriberClient()
        appending_callback_func = appending_callback(bank_list)
        streaming_pull_future = subscriber.subscribe(consumer, callback=appending_callback_func)
        with subscriber:
            try:
                streaming_pull_future.result(timeout=2678400)
            except TimeoutError:
                streaming_pull_future.cancel()  # Trigger the shutdown.
                streaming_pull_future.result()  # Block until the shutdown is complete.


class TopicDictionary(dict):
    """Just a dictionary, but prints relevant warnings for topics"""
    def __setitem__(self, key, item):
        if self.has_key(key):
            print(f"Already a topic named {key}!")
        else:
            self.__dict__[key] = item
    def __getitem__(self, key):
        if self.has_key(key):
            return self.__dict__[key]
        else:
            print(f"No topic named {key}!")
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    def has_key(self, k):
        return k in self.__dict__
    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def pop(self, *args):
        return self.__dict__.pop(*args)
    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)
    def __contains__(self, item):
        return item in self.__dict__
    def __iter__(self):
        return iter(self.__dict__)

