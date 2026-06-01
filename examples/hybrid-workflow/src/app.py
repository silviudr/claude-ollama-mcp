"""A small task tracker API — intentionally incomplete."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Task:
    title: str
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    tags: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None


class TaskStore:
    def __init__(self):
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1

    def add(self, title: str, tags: list[str] | None = None) -> int:
        task_id = self._next_id
        self._tasks[task_id] = Task(title=title, tags=tags or [])
        self._next_id += 1
        return task_id

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def complete(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        if task and not task.is_complete:
            task.completed_at = datetime.now()
            return True
        return False

    def list_all(self) -> list[tuple[int, Task]]:
        return sorted(self._tasks.items())

    # TODO: add filtering by tag
    # TODO: add filtering by completion status
    # TODO: add search by title
    # TODO: add bulk operations
    # TODO: add statistics (completion rate, avg time to complete)
