from tqdm import tqdm

class ProgressBar(tqdm):
    prefixes = []
        
    def add_prefix(self, x):
        self.prefixes.append(x)
        
    def edit_last_prefix(self, x):
        self.prefixes[-1] = x
        
    def clear_prefix(self):
        self.prefixes = []
    
    def set_description(self, x):
        super().set_description(' | '.join(self.prefixes) + ' ' + x)
        
    def __bool__(self):
        return True