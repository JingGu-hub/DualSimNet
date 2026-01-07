import queue

class FixedLengthPriorityQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.pq = queue.PriorityQueue()

    def put(self, item):
        if self.pq.qsize() < self.max_size:
            self.pq.put(item)
        else:
            # 如果队列已满，移除队尾元素
            self.pq.get()
            self.pq.put(item)

    def get(self):
        if not self.pq.empty():
            return self.pq.get()
        else:
            raise queue.Empty("Queue is empty")

    def qsize(self):
        return self.pq.qsize()

    def empty(self):
        return self.pq.empty()

