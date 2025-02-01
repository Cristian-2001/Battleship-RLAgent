import threading

class BlockingTupleSpace:
    def __init__(self):
        self.tuples = []
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def add(self, tuple_data):
        """Add a tuple to the tuple space and notify waiting threads."""
        with self.lock:
            self.tuples.append(tuple_data)
            self.condition.notify_all()  # Notify all waiting threads

    def remove(self, template):
        """Remove and return a tuple that matches the template, blocking until one is available."""
        with self.lock:
            while True:
                for i, tuple_data in enumerate(self.tuples):
                    if self._matches(tuple_data, template):
                        return self.tuples.pop(i)
                # If no matching tuple is found, wait for a notification
                self.condition.wait()

    def read(self, template):
        """Read a tuple that matches the template without removing it, blocking until one is available."""
        with self.lock:
            while True:
                for tuple_data in enumerate(self.tuples):
                    if self._matches(tuple_data, template):
                        return tuple_data
                # If no matching tuple is found, wait for a notification
                self.condition.wait()

    def _matches(self, tuple_data, template):
        """Check if the tuple matches the template."""
        if len(tuple_data) != len(template):
            return False
        for t1, t2 in zip(tuple_data, template):
            if t2 is not None and t1 != t2:
                return False
        return True

# Example usage
if __name__ == "__main__":
    ts = BlockingTupleSpace()

    def producer():
        import time
        time.sleep(2)  # Simulate some delay
        ts.add(("foo", 1))
        print("Producer added ('foo', 1)")
        ts.remove(("foo", "prova", None))
        print("Producer removed ('foo', 'prova', None)")

    def consumer():
        print("Consumer waiting to remove ('foo', None)...")
        result = ts.remove(("foo", None))
        print(f"Consumer removed: {result}")
        ts.add(("foo", "prova", 2))
        print("Consumer added ('foo', 'prova', 2)")

    # Start producer and consumer threads
    threading.Thread(target=producer).start()
    threading.Thread(target=consumer).start()