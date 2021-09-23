from typing import Callable
from concurrent import futures
from google.cloud import pubsub_v1
import threading

__all__ = ["get_callback", "appending_callback"]


def get_callback(
    publish_future: pubsub_v1.publisher.futures.Future, data: str
) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
    def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
        try:
            # Wait 60 seconds for the publish call to succeed.
            print(publish_future.result(timeout=60))
        except futures.TimeoutError:
            print(f"Publishing {data} timed out.")

    return callback


def appending_callback(message_bank):
    def _appending_callback_return(message: pubsub_v1.subscriber.message.Message) -> None:
        message_bank.append(message)
        message.ack()
        if threading.current_thread().stopped():
            threading.current_thread().join()
            raise Exception("Exiting pubsub thread - you shouldn't see this!")
    return _appending_callback_return