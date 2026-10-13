"""Ephemeral active indexes rebuilt once from the durable proposal records.

Every state transition enters/leaves the indexes through persist(). Historical
terminal records never enter the tick path. Heap membership prevents duplicate
admission after retries; rebuilding preserves ordinal proposal order on resume.
"""
import heapq

class ActiveIndex:
    def __init__(self, roots, retryable):
        self.roots = roots
        self.retryable = retryable
        self.states = {}
        self.members = {'pending': set(), 'preparing': set(), 'retry': set()}
        self.heaps = {'pending': [], 'preparing': []}
        for root in roots.values(): self.transition(root)

    def transition(self, root):
        rid = root['root_id']
        previous = self.states.get(rid)
        state = root['state']
        if previous != state:
            if previous in self.members: self.members[previous].discard(rid)
            if state in self.heaps:
                self.members[state].add(rid)
                heapq.heappush(self.heaps[state], (root['_pipeline_ordinal'], rid))
            self.states[rid] = state
        if self.retryable(root): self.members['retry'].add(rid)
        else: self.members['retry'].discard(rid)

    def count(self, state): return len(self.members[state])

    def first(self, state):
        heap = self.heaps[state]
        while heap and heap[0][1] not in self.members[state]: heapq.heappop(heap)
        return self.roots[heap[0][1]] if heap else None

    def reserved_preparation(self, active):
        # Only the bounded preparing set; active tasks, never terminal history.
        candidates = self.members['preparing'] - active
        if not candidates: return None
        return self.roots[min(candidates, key=lambda rid:self.roots[rid]['_pipeline_ordinal'])]
