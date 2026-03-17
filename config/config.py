import yaml
from pathlib import Path
class Config:
    def __init__(self, config_name):
        config_path = Path(__file__).parent/ f"{config_name}.yaml"
        with open(config_path, "r", encoding = 'utf-8') as f:
            self.data = yaml.safe_load(f)

        self._resolve_conflict()
        self._set_attri(self.data)
    def _deep_overwrite_union(self, self_data, other_data):
        """
        receives two dictionary as input
        merge the k, v of self_data and other_data but prioritize self_data
        Args:
            self_data (dict): 
            other_data (dict): 
        """

        self_data = {k: v for k, v in self_data.items() if k != "default"}
        other_data = {k: v for k, v in other_data.items() if k != "default"}
        for k, v in other_data.items():
            if self_data.get(k) is not None:
                if isinstance(v, dict):
                    v = self._deep_overwrite_union(self_data.get(k), other_data.get(k))
                    self_data[k] = v
            else:
                self_data[k] = v
        return self_data
    def _resolve_conflict(self):
        parent_name = self.data.get('default')
        if parent_name is None:
            return
        parent_path = Path(__file__).parent / f"{parent_name}.yaml"
        with open(parent_path, "r", encoding= 'utf-8') as f:
            parent = Config.__new__(Config)
            parent.data = yaml.safe_load(f)

        parent._resolve_conflict()

        self.data = self._deep_overwrite_union(self.data, parent.data)
        
    def _set_attri(self, data:dict):
        for k, v in data.items():
            if isinstance(v, dict):
                sub = Config.__new__(Config)
                sub._set_attri(v)
                setattr(self, k, sub)
            else:
                setattr(self, k, v)
if __name__ == "__main__":
    a = Config('base')
    b = Config('finetune')
    print(f"{a.data=}")
    print(f"{b.data=}")
    print(a.train.weight_decay)
    print(b.train.weight_decay)