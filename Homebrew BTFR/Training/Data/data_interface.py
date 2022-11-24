from typing import Optional, Any

class DataInterface:
    def  __init__(self,\
                n_tasks: int,
                train_transform: Optional[Any] = None,
                test_transform: Optional[Any] = None
                )-> None:
        
        self.n_tasks = n_tasks
        self.tasks_train_data = None
        self.tasks_test_data = None