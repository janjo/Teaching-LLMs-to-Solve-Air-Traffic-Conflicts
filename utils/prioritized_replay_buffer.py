import heapq
import random

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = []

    def add(self, conflict_array, score):
        # Add unique tie-breaker to avoid comparing arrays
        item = (score, id(conflict_array), conflict_array)

        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def sample(self, k):
        # Only return the conflicts, not the scores or IDs
        return [conflict for _, __, conflict in heapq.nlargest(k, self.heap, key=lambda x: x[0])]
    
    def sample_random(self, k):
        # Return k randomly chosen conflicts regardless of score
        return [conflict for _, __, conflict in random.sample(self.heap, k)]