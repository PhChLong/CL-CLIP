import yaml
from pathlib import Path
class Config:
    def __init__(self, config_name):
        config_path = Path(__file__).parent/ f"{config_name}.yaml"
        with open(config_path) as f:
            self.data = yaml.safe_load(f)
        
        self._set_attri(self.data)
    
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
    print( a.data )
    print(b.data)
    print(a.train.lr)
    print(b.train.lr)