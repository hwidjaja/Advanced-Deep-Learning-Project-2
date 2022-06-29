class InfiniteDataGen:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch_count = 0
    
    def generate(self): 
        while True:
            for batch in self.dataloader:
                yield batch
            self.epoch_count += 1
